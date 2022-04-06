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
    
def check_md5(path, md5_hash):
    with open(path, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()

        assert md5_returned==md5_hash, f"{os.path.basename(path)} is corrupted, please download it again"

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


#start for speechcommands 12 classes code
SAMPLE_RATE = 16000
FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
_CHECKSUMS = {
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz":
    "3cd23799cb2bbdec517f1cc028f8d43c",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz":
    "6b74f3901214cb2c2934e98196829835",
}

UNKNOWN = [
 'backward',
 'bed',
 'bird', 
 'cat', 
 'dog', 
 'eight', 
 'five', 
 'follow', 
 'forward',
 'four', 
 'happy', 
 'house', 
 'learn', 
 'marvin', 
 'nine', 
 'one',  
 'seven', 
 'sheila', 
 'six', 
 'three', 
 'tree', 
 'two', 
 'visual', 
 'wow',
 'zero'
]

name2idx = {
    'down':0,
    'go':1,
    'left':2,
    'no':3,
    'off':4,
    'on':5,
    'right':6,
    'stop':7,
    'up':8,
    'yes':9,
    '_silence_':10,
    '_unknown_':11
}

idx2name = {}
for name, idx in name2idx.items():
    idx2name[idx] = name

def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output


def load_speechcommands_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)
    
    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, speaker_id, utterance_number


def caching_data(_walker, path, subset):
    cache = []
    for filepath in tqdm.tqdm(_walker, desc=f'Loading {subset} set'):
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        if label in UNKNOWN: # if the label is not one of the 10 commands, map them to unknown
            label = '_unknown_'
        
        
        speaker, _ = os.path.splitext(filename)
        speaker, _ = os.path.splitext(speaker)
        
        # When loading test_set, there is a folder for _silence_
        if label == '_silence_':
            speaker_id = speaker.split(HASH_DIVIDER)
            utterance_number = -1
        else:
            speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
            utterance_number = int(utterance_number)

        # Load audio     
        audio_samples, rate = torchaudio.load(filepath) # loading audio
        # audio_sample (1, len)
        
        if audio_samples.shape[1] != SAMPLE_RATE:
            pad_length = SAMPLE_RATE-audio_samples.shape[1]
            audio_samples = F.pad(audio_samples, (0,pad_length)) # pad the end of the audio until 1 second
            # (1, 16000)
        cache.append((audio_samples, rate, name2idx[label], speaker_id, utterance_number)) 
    
    
    # include silence
    if subset=='training':
        slience_clips = [
            'dude_miaowing.wav',
            'white_noise.wav',
            'exercise_bike.wav',
            'doing_the_dishes.wav',
            'pink_noise.wav'
        ]
    elif subset=='validation':
        slience_clips = [
            'running_tap.wav'
        ]
    else:
        slience_clips = []
        
        
    for i in slience_clips: 
        audio_samples, rate = torchaudio.load(os.path.join(path, '_background_noise_', i))
        for start in range(0,
                           audio_samples.shape[1] - SAMPLE_RATE,
                           SAMPLE_RATE//2):
            audio_segment = audio_samples[0, start:start + SAMPLE_RATE]
            cache.append((audio_segment.unsqueeze(0), rate, name2idx['_silence_'], '00000000', -1))        
        
    return cache

class SPEECHCOMMANDS_12C(Dataset):
    """Create a Dataset for Speech Commands.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
            (default: ``"speech_commands_v0.02"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"SpeechCommands"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. "validation" and "testing" are defined in "validation_list.txt" and
            "testing_list.txt", respectively, and "training" is the rest. Details for the files
            "validation_list.txt" and "testing_list.txt" are explained in the README of the dataset
            and in the introduction of Section 7 of the original paper and its reference 12. The
            original paper can be found `here <https://arxiv.org/pdf/1804.03209.pdf>`_. (Default: ``None``)
    """

    def __init__(self,
                 root,
                 url,
                 folder_in_archive,
                 download,
                 subset,
                 ):

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            + "{'training', 'validation', 'testing'}."
        )

        if subset in ["training", "validation"]:
            url = "speech_commands_v0.02"
            
        elif subset=='testing':
            url = "speech_commands_test_set_v0.02"
        
        base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
        ext_archive = ".tar.gz"

        url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        print(f"{basename=}")
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive, self._path)

        if subset == "validation":
            self._walker = _load_list(self._path, "validation_list.txt")
            self._data = caching_data(self._walker, self._path, subset)            
        elif subset == "testing":
            self._walker = list(Path(self._path).glob('*/*.wav'))
            self._data = caching_data(self._walker, self._path, subset)
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
            self._walker = [
                w for w in walker
                if HASH_DIVIDER in w
                and EXCEPT_FOLDER not in w
                and os.path.normpath(w) not in excludes
            ]
            self._data = caching_data(self._walker, self._path, subset)
            
        else:
            walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            (Tensor, int, str, str, int):
            ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        return self._data[n]


    def __len__(self) -> int:
        return len(self._data)       



