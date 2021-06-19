from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from typing import Tuple, Union
from torch import Tensor
import torch
import os
import time
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

from torchaudio.compliance import kaldi # for downsampling

_CHECKSUMS = {"mls_italian_opus": "ca5a74d7e97cc62635022719e0ef529d",
    "http://www.openslr.org/resources/12/dev-other.tar.gz":
    "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",
    "http://www.openslr.org/resources/12/test-clean.tar.gz":
    "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "http://www.openslr.org/resources/12/test-other.tar.gz":
    "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz":
    "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz":
    "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
    "http://www.openslr.org/resources/12/train-other-500.tar.gz":
    "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2"
}

class MultilingualLibriSpeech(Dataset):
    """Dataset class for Multilingual LibriSpeech (MLS) dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        language_name (str, optional): Choose the folder corresponding to the language,
            for example ``"mls_english"``. For ``"opus"`` version, simply use ``"mls_english_opus".
            ``(default: ``"mls_german_opus"``).
        split (str): Choose different dataset splits such as ``"train"``, ``"dev"``, and
            ``"test"``. (default: ``"train"``)

        low_resource: bool = False,
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """
    
#     _ext_txt = ".trans.txt"
    _ext_audio = ".opus"
    _ext_pytorch = ".pt"
    

    def __init__(self,
                 root: Union[str, Path],
                 language_name: str = "mls_italian_opus",
                 split: str = "train",                 
                 low_resource: bool = False,
                 one_hr=0,
                 refresh: bool = False,
                 download: bool = False,
                 _ext_txt='.ipa_trans.txt'):

        self._ext_txt = _ext_txt
        self.refresh = refresh
        ext_archive = '.tar.gz'
        url = f"https://dl.fbaipublicfiles.com/mls/{language_name}{ext_archive}"
        
        # Getting audio path
        download_path = os.path.join(root, language_name)
        self._path = os.path.join(root, language_name, split)
        
        if download:
    
            if os.path.isdir(download_path):
                print(f'Dataset exists, skipping download...')
            else:
                checksum = _CHECKSUMS.get(language_name, None)        
                if not os.path.isdir(root):
                    print(f'Creating download path = {root}')
                    os.makedirs(os.path.join(root))
#                 if os.path.isfile(download_path+ext_archive):
#                     print(f'.tar.gz file exists, proceed to extraction...')
#                 else:
                try:
                    download_url(url, root, hash_value=checksum)
                except:
                    print(f'{download_path+ext_archive} already exists, proceed to extraction...')
                extract_archive(download_path+ext_archive)

        
        if os.path.isdir(self._path):
            pass
        else:
            raise FileNotFoundError("Dataset not found, please specify the correct location or set `download=True`")



        if low_resource:
            if one_hr==False and isinstance(one_hr, bool):
                print(f'Using 9hr data at {self._path}')
                with open(os.path.join(self._path, 'limited_supervision', '9hr', 'handles.txt'), 'r') as f:
                    self._walker = f.read().splitlines()
            else:
                print(f'Using 1hr data at {self._path}')
                with open(os.path.join(self._path, 'limited_supervision', '1hr', str(one_hr), 'handles.txt'), 'r') as f:
                    self._walker = f.read().splitlines()                
                    
        else:
            if one_hr:
                raise ValueError(f"When `low_resource`={low_resource}, "\
                                 f"`one_hr` should also be False. But received {one_hr} instead")
            else:
                print(f'Using all data at {self._path}')
                # It seems we don't need to + ._ext_audio
                self._walker = sorted(str(p.stem) for p in Path(self._path).glob('audio/*/*/*' + self._ext_audio))

    def __getitem__(self, n):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return self.load_librispeech_item(fileid, self._path, self._ext_audio)


    def __len__(self) -> int:
        return len(self._walker)
    
    def load_librispeech_item(self, fileid: str,
                              path: str,
                              ext_audio: str)-> Tuple[Tensor, int, str, int, int, int]:
        speaker_id, chapter_id, utterance_id = fileid.split("_")

        file_text = os.path.join(path, 'audio', speaker_id, chapter_id,
                                 speaker_id+'_'+chapter_id+self._ext_txt)

        saved_name = fileid + self._ext_pytorch
        processed_data_path = os.path.join(path, 'audio', speaker_id, chapter_id, saved_name)
        if os.path.exists(processed_data_path) and self.refresh==False:
            return torch.load(processed_data_path)
        else:
#             print(f'MLS file not exists')
            file_audio = fileid + ext_audio
            file_audio = os.path.join(path, 'audio', speaker_id, chapter_id, file_audio)


            # Load audio
            start_audioload = time.time()
            waveform, sample_rate = torchaudio.load(file_audio)
            if sample_rate!=16000: # If the sampling_rate is above 16k, downsample it
#                 print(f'downsampling...')
                waveform = kaldi.resample_waveform(waveform, sample_rate, 16000)

            # Load text
            with open(file_text) as ft:
                for line in ft:
                    fileid_text, utterance = line.strip().split("\t", 1)
                    if '_' in utterance:
                        print(f'file={file_audio} contains _')                    
                    if fileid == fileid_text:
                        break
                else:
                    # Translation not found
                    raise FileNotFoundError("Translation not found for " + fileid_audio)


            batch = {
                     "path": file_audio,
                     "waveform": waveform,
                     "sample_rate": 16000,
                     "utterance": utterance,
                     "speaker_id": int(speaker_id),
                     "chapter_id": int(chapter_id),
                     "utterance_id": int(utterance_id)
                    }

            torch.save(batch, processed_data_path)
        return batch
    

#------------------------Original Dataset from PyTorch----------------
#------------------------I am going to modify it so that it works with IPA labels--------------



