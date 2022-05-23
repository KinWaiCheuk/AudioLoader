from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from typing import Tuple, Union
from torch import Tensor
import torch
import os
import time
import tqdm
import shutil
import glob
import multiprocessing as mp
import warnings
from distutils.dir_util import copy_tree
from torchaudio.compliance import kaldi # for downsampling
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
import hashlib
import torch.nn.functional as F            
from AudioLoader.music.utils import check_md5
             
class TIMIT(Dataset):
    """Dataset class for Multilingual LibriSpeech (MLS) dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        
        language_name (str, optional): 
            Choose the folder corresponding to the language,
            for example ``"mls_english"``. For ``"opus"`` version, simply use ``"mls_english_opus".
            ``(default: ``"mls_german_opus"``).
            
        split (str):
            Choose different dataset splits such as ``"train"``, or ``"test"``. (default: ``"train"``).
            ``"dev"`` is not given in this dataset, you have to split part of the ``"train"`` into the ``"dev"`` set.

        
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
            
        _ext_txt (str, optional):
            Change the file extensions to choose the label type.
            `trans.txt` corresponds to text labels.
            `.ipa_trans.txt` corresponds to IPA labels.
        
    """
    

    def __init__(self,
                 root: Union[str, Path],
                 split: str = "train",                 
                 groups: Union[str, int] = 'all',
                 download: bool = False
                ):

        self._ext_audio = ".wav" # The audio format in this dataset is .wav
        ext_archive = '.zip'
#         url = "https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech/download"
        url = 'https://data.deepai.org/timit.zip'
        
        # Getting audio path
        archive_name = 'timit'
        folder_name = 'TIMIT'
        download_path = os.path.join(root, folder_name)
        self.download_path = download_path
        assert split.upper()=="TRAIN" or split.upper()=="TEST", f"split={split} is not present in this dataset"
        self._path = os.path.join(root, folder_name, 'data', split.upper())
        
        self.groups = self.available_groups(groups)
        
        checksum = '5b736303c55cf4970926bb9978b655fe'
        
        if download:
            if os.path.isdir(download_path) and os.path.isdir(os.path.join(download_path, 'data')):
                print(f'Dataset folder exists, skipping download...')
                decision = input(f"Do you want to extract {archive_name+ext_archive} again? "
                                 f"To avoid this prompt, set `download=False`\n"
                                 f"This action will overwrite exsiting files, do you still want to continue? [yes/no]") 
                if decision.lower()=='yes':
                    print(f'extracting...')
                    extract_archive(os.path.join(download_path, archive_name+ext_archive))                
            elif os.path.isfile(os.path.join(download_path, 'timit.zip')):
                print(f'timit.zip exists, extracting...')
                check_md5(os.path.join(download_path, archive_name+ext_archive), checksum)
                extract_archive(os.path.join(download_path, archive_name+ext_archive))
            else:
                decision='yes'       
                if not os.path.isdir(download_path):
                    print(f'Creating download path = {root}')
                    os.makedirs(os.path.join(download_path))
#                 if os.path.isfile(download_path+ext_archive):
#                     print(f'.tar.gz file exists, proceed to extraction...')
#                 else:
                if os.path.isfile(os.path.join(download_path, archive_name+ext_archive)):
                    print(f'{download_path+ext_archive} already exists, proceed to extraction...')
                else:
                    print(f'downloading...')
                    try:
                        download_url(url, download_path, hash_value=checksum, hash_type='md5')
                    except:
                        raise Exception('Auto download fails. '+
                                        'You may want to download it manually from:\n'+
                                        url+ '\n' +
                                        f'Then, put it inside {download_path}')

        
        if os.path.isdir(self._path):
            pass
        elif os.path.isfile(os.path.join(download_path, 'timit.zip')):
            print(f'timit.zip exists, extracting...')
            check_md5(os.path.join(download_path, archive_name+ext_archive), checksum)
            extract_archive(os.path.join(download_path, archive_name+ext_archive))
            
        else:
            raise FileNotFoundError(f"Dataset not found at {self._path}, please specify the correct location or set `download=True`")
                    
        print(f'Using all data at {self._path}')
        # It seems we don't need to + ._ext_audio
        self._walker = []
        for group in self.groups:
            for file in [str(p) for p in Path(self._path).glob(os.path.join(group, '*', '*.wav'))]:
                self._walker.append(file)
                

    def __getitem__(self, n):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        file_path = self._walker[n]  
        
        # retreiving DR, gender, speakerid and wav_name
        path_split = file_path.split(os.path.sep)
        DR, speaker, wav_name = path_split[-3:]
        gender = speaker[0]
        speaker_id = speaker[1:]
        
        waveform, sample_rate = torchaudio.load(file_path)
        
        phonemics = self.read_labels(file_path, 'PHN')
        words = self.read_labels(file_path, 'WRD')
        
        batch = {'path': file_path,
                 'waveform': waveform,
                 'sample_rate': sample_rate,
                 'DR': DR,
                 'gender': gender,
                 'speaker_id': speaker_id,
                 'phonemics': phonemics,
                 'words': words
                }
        
        return batch

    
    def read_labels(self, file_path, ext):
        """
        ext: either `'WRD` for word labels or `PHN` for phone labels
        """
        label_path = file_path.replace('WAV.wav', ext)
        labels = ''
        with open(label_path, 'r') as f:
            lines = f.read().splitlines()
            for i in lines:
                labels += f"{(i.split(' ')[-1])} "
            if ext=='PHN':
                # Remove the slience indicator h# at the beginning and the end
                labels = labels.replace('h# ', '')
                
        return labels
    
    def __len__(self) -> int:
        return len(self._walker)
    
            
    def available_groups(self, groups):
        if groups=='all':
            return [f'DR{i+1}' for i in range(9)] # select all dialect regions
        elif isinstance(groups, list):
            return [f'DR{i}' for i in groups]   