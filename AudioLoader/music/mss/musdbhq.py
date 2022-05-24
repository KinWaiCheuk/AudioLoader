import json
import os
from pathlib import Path
from abc import abstractmethod
from glob import glob
import csv
import shutil
import sys
import pickle
import numpy as np
import random
import yaml
from typing import Optional, Callable
# import soundfile
from tqdm import tqdm
import multiprocessing
import musdb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from torchaudio.compliance import kaldi # for downsampling
import hashlib
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

from collections import OrderedDict
import math

metadata= './metadata'
sources= ['drums', 'bass', 'other', 'vocals']
MIXTURE = 'mixture'
EXT = '.wav'

class MusdbHQ:
    def __init__(
            self,root, subset,download = False,segment=None, shift=None, normalize=True,
            samplerate=44100, channels=2, ext=EXT):
        """
        MusdbHQ (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.
        Args:
            root (Path or str): root folder for the dataset.
            subset (str): training ,validation, traininf_all or test
            download (bool): Whether to download the dataset if it is not found at root path. (default: ``False``).
            segment (None or float): segment length in seconds. If `None`, returns entire tracks.
            shift (None or float): stride in seconds bewteen samples.
            normalize (bool): normalizes input audio, **based on the metadata content**,
                i.e. the entire track is normalized, not individual extracts.
            samplerate (int): target sample rate. if the file sample rate
                is different, it will be resampled on the fly.
            channels (int): target nb of channels. if different, will be
                changed onthe fly.
            ext (str): extension for audio files (default is .wav).
        samplerate and channels are converted on the fly.
        """
 
         # Getting audio path
        ext_archive = '.zip'
        archive_name = 'musdb18hq'

        download_path = Path(root)
        self.download_path = download_path
               
        if subset.lower() == 'training' or subset.lower() == 'validation' or subset.lower() == 'training_all':
            dataset = 'train'
        elif subset.lower() == 'test':
            dataset = 'test'
        else:
            print(f'Subset does not exist, please choose from training, validation, training_all or test')
        
    
        self._path = os.path.join(root, dataset.lower())

        url= 'https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1'
        checksum = '12d4f2ecd55245a4688754dd76363103'
        
        if download:
            if os.path.isdir(download_path) and os.path.isdir(os.path.join(download_path, 'train')) and os.path.isdir(os.path.join(download_path, 'test')):
                print(f'Dataset folder exists, skipping download...')
                decision = input(f"Do you want to extract {archive_name+ext_archive} again? "
                                 f"To avoid this prompt, set `download=False`\n"
                                 f"This action will overwrite exsiting files, do you still want to continue? [yes/no]") 
                if decision.lower()=='yes':
                    print(f'extracting...')
                    extract_archive(os.path.join(download_path, archive_name+ext_archive))                
            elif os.path.isfile(os.path.join(download_path, 'musdb18hq.zip')):
                print(f'musdb18hq.zip exists, checking MD5...')
                check_md5(os.path.join(download_path, archive_name+ext_archive), checksum)
                print(f'MD5 is correct, extracting...')
                extract_archive(os.path.join(download_path, archive_name+ext_archive))
            else:
                decision='yes'       
                if not os.path.isdir(download_path):
                    print(f'Creating download path = {root}')
                    os.makedirs(os.path.join(download_path))

                if os.path.isfile(os.path.join(download_path, archive_name+ext_archive)):
                    print(f'{download_path+archive_name+ext_archive} already exists, proceed to extraction...')
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
        elif os.path.isfile(os.path.join(download_path, 'musdb18hq.zip')):
            print(f'musdb18hq.zip exists, checking MD5...')
            check_md5(os.path.join(download_path, archive_name+ext_archive), checksum)
            print(f'MD5 is correct, extracting...')            
            extract_archive(os.path.join(download_path, archive_name+ext_archive))
            
        else:
            raise FileNotFoundError(f"Dataset not found at {self._path}, please specify the correct location or set `download=True`")
                    
        print(f'Using data at {self._path}')
        # It seems we don't need to + ._ext_audio
        self._walker = []
                
# get_musdb_wav_datasets             
        if subset.lower() == 'training' or subset.lower() == 'validation' or subset.lower() == 'training_all':
            root = Path(root) / "train"
            metadata_file = Path('./metadata') / ('musdb_' + 'train' + ".json")
        
        elif subset.lower() == 'test':
            root = Path(root) / "test"
            metadata_file = Path('./metadata') / ('musdb_' + 'test' + ".json")
        
    #     if not metadata_file.is_file() and distrib.rank == 0:
        if not metadata_file.is_file():
            metadata_file.parent.mkdir(exist_ok=True, parents=True)
            metadata = build_metadata(root, sources)
            json.dump(metadata, open(metadata_file, "w"))
    #     if distrib.world_size > 1:
    #         distributed.barrier()
        metadata = json.load(open(metadata_file))

        valid_tracks = _get_musdb_valid()
        
        if subset.lower() == 'training':
            metadata = {name: meta for name, meta in metadata.items() if name not in valid_tracks}
            self.sources = sources
            
        elif subset.lower() == 'validation':
            metadata = {name: meta for name, meta in metadata.items() if name in valid_tracks}
            self.sources = [MIXTURE] + list(sources)
            
        elif subset.lower() == 'training_all':
            metadata_train = metadata
            self.sources = sources
        
        elif subset.lower() == 'test':
            metadata_test = metadata
            self.sources = [MIXTURE] + list(sources)    
            
# metadata (dict): output from `build_metadata`.
# sources (list[str]): list of source names.
        
        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_duration = meta['length'] / meta['samplerate']
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - self.segment) / self.shift) + 1)
            self.num_examples.append(examples)
# samplerate = number of sample per second
# length = number of sample
# track_duration is second, cut song into segment

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
#         print(len(self.metadata))
        for name, examples in zip(self.metadata, self.num_examples):           
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))
            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = torchaudio.load(str(file), frame_offset=offset, num_frames=num_frames)
                wav = convert_audio_channels(wav, self.channels)
                wavs.append(wav)

            example = torch.stack(wavs)
            example = torchaudio.functional.resample(example, meta['samplerate'], self.samplerate)
            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def build_metadata(path, sources, normalize=True, ext=EXT):
    """
    Build the metadata for `MusdbHQ`.
    Args:
        path (str or Path): path to dataset.
        sources (list[str]): list of sources to look for.
        normalize (bool): if True, loads full track and store normalization
            values based on the mixture file.
        ext (str): extension of audio files (default is .wav).
    """

    meta = {}
    path = Path(path)
    pendings = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        for root, folders, files in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith('.') or folders or root == path:
                continue
            name = str(root.relative_to(path))
            pendings.append((name, pool.submit(_track_metadata, root, sources, normalize, ext)))
            # meta[name] = _track_metadata(root, sources, normalize, ext)
        for name, pending in tqdm(pendings, ncols=120):
            meta[name] = pending.result()
    return meta



def _track_metadata(track, sources, normalize=True, ext=EXT):
    track_length = None
    track_samplerate = None
    mean = 0
    std = 1
    for source in sources + [MIXTURE]:
        file = track / f"{source}{ext}"
        try:
            info = torchaudio.info(str(file))
        except RuntimeError:
            print(file)
            raise
        length = info.num_frames
        if track_length is None:
            track_length = length
            track_samplerate = info.sample_rate
        elif track_length != length:
            raise ValueError(
                f"Invalid length for file {file}: "
                f"expecting {track_length} but got {length}.")
        elif info.sample_rate != track_samplerate:
            raise ValueError(
                f"Invalid sample rate for file {file}: "
                f"expecting {track_samplerate} but got {info.sample_rate}.")
        if source == MIXTURE and normalize:
            try:
                wav, _ = torchaudio.load(str(file))
            except RuntimeError:
                print(file)
                raise
            wav = wav.mean(0)
            mean = wav.mean().item()
            std = wav.std().item()

    return {"length": length, "mean": mean, "std": std, "samplerate": track_samplerate}


def _get_musdb_valid():
    # Return musdb valid set.
    import yaml
    setup_path = Path(musdb.__path__[0]) / 'configs' / 'mus.yaml'
    setup = yaml.safe_load(open(setup_path, 'r'))
    return setup['validation_tracks']
        
        
def check_md5(path, md5_hash):
    """
    This version of cehck_md5 reads file chunk by chunk to avoid memory error (file size > RAM)
    """
    md5 = hashlib.md5()
    with open(path,'rb') as f: 
        for chunk in iter(lambda: f.read(8192), b''): 
            md5.update(chunk)
    md5_returned = md5.hexdigest()
    assert md5_returned==md5_hash, f"{os.path.basename(path)} is corrupted, please download it again"
