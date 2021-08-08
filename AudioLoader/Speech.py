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



_CHECKSUMS = {"mls_english_opus": "60390221eec6f456611563b37f0b052c",
              "mls_german_opus":  "b24a9dfd3a8dd1aeabc1341982cc4775",
              "mls_dutch_opus":   "96658c55ef85993a56cf2efbf6f83f57",
              "mls_french_opus":  "b4716b2fd4f67b07c17c6b2a6068641f",
              "mls_spanish_opus": "a2c4b0ff02a71f10ddb21beffbe640ce",
              "mls_italian_opus": "ca5a74d7e97cc62635022719e0ef529d",
              "mls_portuguese_opus": "4dbd6cbdda61268e5d26c4117b0bf769",
              "mls_polish_opus":  "21f83647876c61566c96fdc6298a7b65",
             
              "mls_english": "9d4249911e318c2b8dcfcfecb484d865",
              "mls_german": "91ac982bf63869307f1b8950dfb7c776",
              "mls_dutch": "6b171a16baff0108efd320b1ad65b9d1",
              "mls_french": "4172e807697259bff9ad63661aecabf6",
              "mls_spanish": "6c34698dd522dde28fdc43309e9cc1ac",
              "mls_italian": "dc77f5805aecc7182aa20786032e5dc1",
              "mls_portuguese": "12d54613fae75ae5fb1d55836408f3ee",
              "mls_polish": "ce1a1278006cc373c9d1cb6dbfc03d47"}


espeak_map = {"mls_english_opus": "en",
              "mls_german_opus":  "de",
              "mls_dutch_opus":   "nl",
              "mls_french_opus":  "fr-fr",
              "mls_spanish_opus": "es",
              "mls_italian_opus": "it",
              "mls_portuguese_opus": "pt-pt",
              "mls_polish_opus":  "pl",
              "mls_english": "en",
              "mls_german":  "de",
              "mls_dutch":   "nl",
              "mls_french":  "fr",
              "mls_spanish": "es",
              "mls_italian": "it",
              "mls_portuguese": "pt-pt",
              "mls_polish":  "pl"}
    


class MultilingualLibriSpeech(Dataset):
    """Dataset class for Multilingual LibriSpeech (MLS) dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        
        language_name (str, optional): 
            Choose the folder corresponding to the language,
            for example ``"mls_english"``. For ``"opus"`` version, simply use ``"mls_english_opus".
            ``(default: ``"mls_german_opus"``).
            
        split (str):
            Choose different dataset splits such as ``"train"``, ``"dev"``, and
            ``"test"``. (default: ``"train"``)

        low_resource: bool = False,
        
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
            
        _ext_txt (str, optional):
            Change the file extensions to choose the label type.
            `trans.txt` corresponds to text labels.
            `.ipa_trans.txt` corresponds to IPA labels.
        
    """
    

    def __init__(self,
                 root: Union[str, Path],
                 language_name: str = "mls_italian_opus",
                 split: str = "train",                 
                 low_resource: bool = False,
                 one_hr=0,
                 use_cache: bool = False,
                 refresh: bool = False,
                 sample_rate: int = None,
                 download: bool = False,
                 IPA: bool = False,
                 _ext_txt='.trans.txt'):
        if 'opus' in language_name:
            self._ext_audio = ".opus"
        else:
            self._ext_audio = ".flac"
        self.espeak_map = espeak_map[language_name]
        print(f'using {self.espeak_map} as espeak language')
        self._ext_pytorch = ".pt"
        self._ext_txt = _ext_txt
        self.use_cache = use_cache
        ext_archive = '.tar.gz'
        self.sample_rate = sample_rate
        url = f"https://dl.fbaipublicfiles.com/mls/{language_name}{ext_archive}"
        
        # Getting audio path
        download_path = os.path.join(root, language_name)
        self.download_path = download_path
        self._path = os.path.join(root, language_name, split)
        
        if download:
    
            if os.path.isdir(download_path):
                print(f'Dataset folder exists, skipping download...')
                decision = input(f"Do you want to extract {language_name+ext_archive} again?\n"+
                                 f"This action will overwrite exsiting files, do you still want to continue? [yes/no]")                
            else:
                decision='yes'
                checksum = _CHECKSUMS.get(language_name, None)        
                if not os.path.isdir(root):
                    print(f'Creating download path = {root}')
                    os.makedirs(os.path.join(root))
#                 if os.path.isfile(download_path+ext_archive):
#                     print(f'.tar.gz file exists, proceed to extraction...')
#                 else:
                if os.path.isfile(os.path.join(root, language_name+ext_archive)):
                    print(f'{download_path+ext_archive} already exists, proceed to extraction...')
                else:
                    print(f'downloading...')
                    download_url(url, root, hash_value=checksum, hash_type='md5')
                    
            if decision.lower()=='yes':
                print(f'extracting...')
                extract_archive(download_path+ext_archive)


            print(f'Splitting utterance labels, it might take a long time for large languages such as English.')
#             thread_num = input(f"How many threads do you want to use?\n"+
#                                f"[If you want to use single-thread, enter 0]")
#             thread_num = int(thread_num)
            self.extract_labels('train', num_threads=0, IPA=False)
            self.extract_labels('dev', num_threads=0, IPA=False)
            self.extract_labels('test', num_threads=0, IPA=False)
        
        if os.path.isdir(self._path):
            pass
        else:
            raise FileNotFoundError(f"Dataset not found at {self._path}, please specify the correct location or set `download=True`")



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
                
        if refresh==True:
            if self.sample_rate:
                print(f'Resampling audio files to {self.sample_rate}Hz. '
                      f'You may change the `sample_rate` argument in the dataset class if you want')
            else:
                print(f'No sample_rate is given, audio files will not be resampled')                
            self.clear_cache()

    def __getitem__(self, n):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        return self.load_librispeech_item(fileid, self._path, self._ext_audio)
        

    def extract_limited_train_set(self):
        root = os.path.join(self.download_path, 'train', 'limited_supervision')
        audio_root = os.path.join(self.download_path, 'train', 'audio')
        target_root = os.path.join(self.download_path, 'limited_train', 'audio')
        
        if os.path.exists(target_root):
            decision = input(f"{target_root} already exists, this function will overwrite existing files.\n"+
                             f"Do you still want to continue? [yes/no]")
            if decision.lower() == 'no':
                print(f"Audio extraction aborted.")
                return
            elif decision.lower() == 'yes':
                pass
            else:
                raise ValueError(f"'{decision}' is not a valid answer")
        
        # read the files for the 9hr set
        with open(os.path.join(root, '9hr', 'handles.txt')) as f:
            limited_9hr_paths = f.read().splitlines()
        # read the files for the 1hr set
        limited_1hr_paths = {} # create a dictionary to sort the k-fold 1hr set
        for folder_name in range(6):
            # read the files for the 1hr set
            with open(os.path.join(root, '1hr', str(folder_name), 'handles.txt')) as f:
                limited_1hr_paths[folder_name] = f.read().splitlines()

        # Appending 9hr and 1hr set
        unique_paths = set()
        # Add 9hr sets to the unique set
        unique_paths.update(set(limited_9hr_paths))
        # Add the k-fold of 1hr set to the unique set
        for key, values in limited_1hr_paths.items():
            unique_paths.update(set(values))
        
        # Moving the audio files from limited set into a new folder
        for i in tqdm.tqdm(unique_paths, desc='Creating `limited_train` set'):
            speaker_id, chapter_id, utterance_id = i.split('_')
            audio_path = os.path.join(audio_root, speaker_id, chapter_id,
                                      i+self._ext_audio)

            output_path = os.path.join(target_root, speaker_id, chapter_id,
                                       i+self._ext_audio)
            output_folder = os.path.dirname(output_path)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            shutil.copy(audio_path, output_path)
            
        # Moving the rest of the files
        train_folder = os.path.join(*audio_root.split('/')[:-1])
        target_train_folder = os.path.join(*target_root.split('/')[:-1])
        # moving metadata inside the `limited_supervision` folder
        copy_tree(os.path.join(train_folder, 'limited_supervision'), os.path.join(target_train_folder, 'limited_supervision'))        
        # moving all .txt files
        for i in glob.glob(os.path.join(train_folder, '*.txt')):
            try:
                shutil.copy(i, target_train_folder)
            except Exception as e:
                pass        
        
    def extract_labels(self, split_name, num_threads=0, Text=True, IPA=False):        
        root = os.path.join(self.download_path, split_name)
        # check if IPA labels already exist
        if IPA:
            ipa_labels = glob.glob(os.path.join(root, 'audio', '*','*','*.ipa_trans.txt'))
            if len(ipa_labels)>0:
                decision = input(f'IPA labels `ipa_trans.txt` in {split_name} folder already exist,'
                                 f'do you want to extract `.ipa_trans.txt` labels again? [yes/no]\n'
                                 f'Warning: All the `.ipa_trans.txt` files inside {split_name} folder will be removed.')
                if decision.lower() == 'no':
                    IPA=False
                elif decision.lower() == 'yes':
                    for i in ipa_labels:
                        os.remove(i)
                else:
                    raise ValueError(f"'{decision}' is not a valid answer")
                    
                    
        labels = glob.glob(os.path.join(root, 'audio', '*','*','*.trans.txt'))
        if len(labels)>0:
            decision = input(f'text labels `.trans.txt` in {split_name} folder already exist,'
                             f'do you want to extract `.trans.txt` labels again? [yes/no]\n'
                             f'Warning: All the `.trans.txt` files inside {split_name} folder will be removed.')
            if decision.lower() == 'no':
                Text=False
                if IPA==False and Text==False:
                    return
            elif decision.lower() == 'yes':
                for i in labels:
                    os.remove(i)
            else:
                raise ValueError(f"'{decision}' is not a valid answer")                    
        
        # Loading all labels into a dictionary
        label_path = os.path.join(root, 'transcripts.txt')
        labels = {}
        with open(label_path,'r') as input_file:
            lines = input_file.readlines()
            for line in tqdm.tqdm(lines, desc='Reading transcripts.txt'):
                utterance_id, text = line.split('\t')
                labels[utterance_id] = text

        # Split labels when the audio exists        
        audio_list = glob.glob(os.path.join(root, 'audio', '*','*',f'*{self._ext_audio}'))
        
        if num_threads==0:
            for audio_file in tqdm.tqdm(audio_list, desc=f"Extracting {split_name} set"):
                self._write_labels(audio_file,labels,root, Text, IPA)
        elif num_threads>0: # use multithread
            pool = mp.Pool(num_threads)
            for audio_file in tqdm.tqdm(audio_list, desc=f"Extracting {split_name} set"):
                r = pool.apply_async(self._write_labels, args=(audio_file,labels,root,Text,IPA,))
    #             r.get()
            pool.close()
            pool.join()
        else:
            raise ValueError(f"'num_threads={num_threads}' is not a valid value")
            
        
    
    def __len__(self) -> int:
        return len(self._walker)
    
    def clear_cache(self):
        """
        This method removes all the .pt file in the dataset.
        By removing all the .pt files, ``__getitem__`` will downsample and
        save the .pt files again when ``use_cache=True``."""
        
        cache_files = glob.glob(os.path.join(self._path,'*','*','*','*.pt'))
        if len(cache_files)>0:
            decision = input(f'{len(cache_files)} .pt files found, confirm clearing cache? [yes/no]')
            if decision.lower()=='yes':
                for i in cache_files:
                    os.remove(i)
            elif decision.lower()=='no':
                print(f'Aborting cache cleaning...')
            else:
                raise ValueError(f'Input {decision} is not recognized, please choose `yes` or `no`.')
        elif len(cache_files)==0:
            print(f'No cache file found')
        else:
            raise ValueError(f'Something is wrong, why does there are {len(cache_files)} cache files')
    
    def load_librispeech_item(self, fileid: str,
                              path: str,
                              ext_audio: str)-> Tuple[Tensor, int, str, int, int, int]:
        speaker_id, chapter_id, utterance_id = fileid.split("_")

        file_text = os.path.join(path, 'audio', speaker_id, chapter_id,
                                 speaker_id+'_'+chapter_id+self._ext_txt)

        saved_name = fileid + self._ext_pytorch
        processed_data_path = os.path.join(path, 'audio', speaker_id, chapter_id, saved_name)
        
        # TODO: check if .pt file exists, if not, move to else
        if self.use_cache==True and os.path.isfile(processed_data_path):
            return torch.load(processed_data_path)
        else:
#             print(f'MLS file not exists')
            file_audio = fileid + ext_audio
            file_audio = os.path.join(path, 'audio', speaker_id, chapter_id, file_audio)


            # Load audio
            start_audioload = time.time()
            waveform, sample_rate = torchaudio.load(file_audio)
            if sample_rate!=self.sample_rate and self.sample_rate!=None: # If the sample_rate is above 16k, downsample it
                waveform = kaldi.resample_waveform(waveform, sample_rate, self.sample_rate)
                sample_rate=self.sample_rate
                
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
                     "sample_rate": sample_rate,
                     "utterance": utterance,
                     "speaker_id": int(speaker_id),
                     "chapter_id": int(chapter_id),
                     "utterance_id": int(utterance_id)
                    }
            if self.use_cache==True:
                torch.save(batch, processed_data_path)
        return batch
    
    def _write_labels(self, audio_file, labels, root, Text, IPA):
        label_id = os.path.basename(audio_file)[:-5]
        speaker_id, chapter_id, _ = label_id.split('_')
        utterance = labels[label_id]          

        if IPA:
            from phonemizer import phonemize, separator  
            ipa_sequence = phonemize(utterance.lower(),
                                     language=self.espeak_map,
                                     backend='espeak',
                                     strip=True,
                                     language_switch='remove-flags',
                                     separator=separator.Separator(phone=" ", word=" <SPACE> "))

            output_file = os.path.join(root, 'audio', speaker_id, chapter_id,
                                       speaker_id + '_' + chapter_id + '.ipa_trans.txt')
            # writing the ipa labels into each chapter
            ipa_output = label_id + '\t' + ipa_sequence

            with open(output_file, 'a') as f:    
                f.write(ipa_output + '\n')


        if Text:
            # writing the text labels into each chapter        
            output_file = os.path.join(root, 'audio', speaker_id, chapter_id,
                                       speaker_id + '_' + chapter_id + '.trans.txt')

            txt_output = label_id + '\t' + utterance
            with open(output_file, 'a') as f:    
                f.write(txt_output)   
            
             
class TIMIT(Dataset):
    """Dataset class for Multilingual LibriSpeech (MLS) dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        
        language_name (str, optional): 
            Choose the folder corresponding to the language,
            for example ``"mls_english"``. For ``"opus"`` version, simply use ``"mls_english_opus".
            ``(default: ``"mls_german_opus"``).
            
        split (str):
            Choose different dataset splits such as ``"train"``, ``"dev"``, and
            ``"test"``. (default: ``"train"``)

        low_resource: bool = False,
        
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
        url = "https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech/download"
        
        # Getting audio path
        archive_name = 'archive'
        folder_name = 'TIMIT'
        download_path = os.path.join(root, folder_name)
        self.download_path = download_path
        self._path = os.path.join(root, folder_name, 'data', split.upper())
        
        self.groups = self.available_groups(groups)
        
        checksum = '8acbaea404859a04925b5bd32474d3ce'
        
        if download:
            if os.path.isdir(download_path) and os.path.isdir(os.path.join(download_path, 'data')):
                print(f'Dataset folder exists, skipping download...')
                decision = input(f"Do you want to extract {archive_name+ext_archive} again? "
                                 f"To avoid this prompt, set `download=False`\n"
                                 f"This action will overwrite exsiting files, do you still want to continue? [yes/no]")                
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
                    
            if decision.lower()=='yes':
                print(f'extracting...')
                extract_archive(os.path.join(download_path, archive_name+ext_archive))

        
        if os.path.isdir(self._path):
            pass
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
        labels = []
        with open(label_path, 'r') as f:
            lines = f.read().splitlines()
            for i in lines:
                labels.append(i.split(' ')[-1]) 
                
        return labels
    
    def __len__(self) -> int:
        return len(self._walker)
    
            
    def available_groups(self, groups):
        if groups=='all':
            return [f'DR{i+1}' for i in range(9)] # select all dialect regions
        elif isinstance(groups, list):
            return [f'DR{i}' for i in groups]           




