import json
import os
from pathlib import Path
from abc import abstractmethod
from glob import glob
import csv
import shutil
import sys
import pickle
import time
import urllib
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

__TORCH_GTE_2_0 = False
split_version = torch.__version__.split(".")
major_version = int(split_version[0])
if major_version > 1:
    __TORCH_GTE_2_0 = True
    from torchaudio.datasets.utils import _extract_zip as extract_archive
    from torch.hub import download_url_to_file as download_url
else:
    from torchaudio.datasets.utils import (
        download_url,
        extract_archive,
    )

from collections import OrderedDict
import math

metadata= './metadata'
MIXTURE = 'mixture'
EXT = '.wav'

"""
Next step: Build a CovNet. 1 channel to 4 channels
Use pytorch lightning template
"""

class Moisesdb23:
    def __init__(
            self,root, download = False, segment=None, shift=None, normalize=True,
            samplerate=16000, channels=2, ext=EXT):
        """
        return mix and sources
        mix: (1, sample len)
        sources: (4, sample len)
        """
        
        self.samplerate = samplerate
        self.root = Path(root)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext        
 
        self.dataset_path = Path(root)
        self._path = list(self.dataset_path.iterdir())
        
        
        # When publishing, use path = Path(__file__).parent.joinpath("MIDI_program_map.tsv") 
        with open(Path(__file__).parent.joinpath(f"moisesdb23_meta_{samplerate}.pkl"), 'rb') as f:
            self._meta = pickle.load(f)
            
        if segment:
            self.segment_samples = int(self.samplerate * segment)
            
    def download(root, dataset='label_noise'):
        """
        dataset: str Either 'label_noise' or 'bleeding'sß
        """
        
        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\rDownloading...%d%%, %d MB, %d KB/s, %d seconds passed" %
                            (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()
        
        if dataset == 'label_noise':
            url = 'https://sdx-2023-data-bucket.s3.amazonaws.com/music-demixing/sony/labelnoise/v1.0-rc1/sdxdb23_labelnoise_v1.0_rc1.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT4DJUO6QYWYWO542%2F20230522%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230522T093442Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e1e5c5f761af681d094f75496baac62f3159fc8fd900457929e005b53bbb174d'
            filename = 'sdxdb23_labelnoise_v1.0_rc1.zip'
        elif dataset == 'bleeding':
            url = 'https://sdx-2023-data-bucket.s3.amazonaws.com/music-demixing/sony/bleeding/v1.0-rc1/sdxdb23_bleeding_v1.0_rc1.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT4DJUO6QYWYWO542%2F20230522%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230522T093442Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2043acf0bb22a38aff3368cf0acf99544f02151343db2e138ad846018cb2d68d'
            filename = 'sdxdb23_bleeding_v1.0_rc1.zip'
        else:
            raise ValueError(f"dataset={dataset} is not a valid option. Please choose between 'label_noise' and 'bleeding'")
            
        urllib.request.urlretrieve(url,
                                   os.path.join(root, filename),
                                   reporthook)
        
        

# samplerate = number of sample per second
# length = number of sample
# track_duration is second, cut song into segment

    def __len__(self):
        return len(self._path)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
        folder_name = self._path[index].stem
        audio_length = self._meta[folder_name]
        
        sources = torch.empty(4, 2, self.segment_samples)
        
        if self.segment_samples:
            assert (audio_length - self.segment_samples)>0, \
            f"segment_samples={self.segment_samples} is longer than the "
            f"audio_length={audio_length}. Please reduce the segment_samples"
            start_sample = np.random.randint(audio_length - self.segment_samples)

            start_time = start_sample/self.samplerate
            # end_sample = start_sample + self.segment_samples

            for idx, source_name in enumerate(['bass', 'drums', 'other', 'vocals']):
                path = os.path.join(self._path[index], source_name + '.flac')
                wav, sr = torchaudio.load(path,
                                          frame_offset=start_sample,
                                          num_frames=self.segment_samples) # (1, L) mono audio            
                    
                # wavs.append(wav) # list won't work in dataloader
                sources[idx] = wav
            
        mix = sources.sum(0, keepdim=True) # creating a max
        return mix, sources


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
