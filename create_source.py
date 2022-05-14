import argparse
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import yaml
import glob
import pickle
import numpy as np

from AudioLoader.music.mss.MIDI_program_map import idx2instrument_class
import torchaudio
import torchaudio.functional as F
import multiprocessing
import joblib
from joblib import Parallel, delayed
# from create_slakh2100 import load_midi_track_group_info_plugin
import sys
import contextlib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def pack_audio_clips(
    input_dir: str,
    output_dir: str,
    sample_rate: int,
    num_workers=-1
    ):
    """
    Pack and resample audio clips into sources

    input_dir: location of Slack2100 dataset
    output_dir: location of the output packed audio
    sample_rate: the sample rate of the output audio

    Returns:
        None
    """

    for split in ["train", "test", "validation"]:
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        split_input_dir = os.path.join(input_dir, split)
        audio_names = sorted(os.listdir(split_input_dir))

#         print("------ Split: {} (Total: {} clips) ------".format(split, len(audio_names)))

        params = []
        for audio_name in audio_names:
            audio_path = os.path.join(split_input_dir, audio_name, "mix.flac")
            output_path = os.path.join(split_output_dir, audio_name)
            os.makedirs(output_path, exist_ok=True)

            param = (audio_path, output_path, audio_name, split, sample_rate)
            # E.g., (0, './datasets/dataset-slakh2100/slakh2100_flac/train/Track00001/mix.flac',
            # './workspaces/hdf5s/waveforms/train/Track00001.h5', 'Track00001', 'train', 16000)
            params.append(param)
        # Debug by uncomment the following code.
        # write_single_audio_to_hdf5(params[0])

        # Pack audio files to hdf5 files in parallel.
#         with ProcessPoolExecutor(max_workers=None) as pool:
#             pool.map(write_audio, params)
        with tqdm_joblib(tqdm(desc=f"Packing {split} set audio clips", total=len(params))) as progress_bar:
                Parallel(n_jobs=num_workers)\
                        (delayed(write_audio)(param) for param in params)     



def write_audio(param):
    r"""Write a single audio file into an hdf5 file.

    Args:
        param: (audio_index, audio_path, output_path, audio_name, split, sample_rate)

    Returns:
        None
    """
    
    [audio_path, output_path, audio_name, split, sample_rate] = param
    audio, sr = torchaudio.load(audio_path)
    audio = F.resample(audio.squeeze(0), sr, sample_rate)

    duration = len(audio) / sample_rate

    torchaudio.save(os.path.join(output_path, 'waveform.flac'),
                    audio.unsqueeze(0),
                    sample_rate)
    
    dirname = os.path.dirname(audio_path) # getting the folder for the audio
    with open(os.path.join(dirname, "metadata.yaml"), "r") as stream:
        stem_dict = yaml.safe_load(stream)['stems']

    source_tracks = {}        
    for source_key, item in stem_dict.items():
        if item['midi_saved'] and item['audio_rendered']: # When midi_save=False, there is no audio track
            source_name = idx2instrument_class[item['program_num']]
            audio, _ = torchaudio.load(os.path.join(dirname, 'stems', f"{source_key}.flac"))
            audio = F.resample(audio.squeeze(0), sr, sample_rate)
#             audio = audio.numpy()

            if source_name in source_tracks.keys():
                source_tracks[source_name] += audio
            else:              
                source_tracks[source_name] = audio
                
    for key, i in source_tracks.items():
        torchaudio.save(
            os.path.join(output_path, f'{key}.flac'),
            i.unsqueeze(0),
            sample_rate
        )