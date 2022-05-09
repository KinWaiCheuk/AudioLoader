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




