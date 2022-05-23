import torch
import numpy as np
import time

def tsv2roll(tsv, audio_length, sample_rate, hop_size, max_midi, min_midi):
    """
    Converting a tsv file into a piano roll
    
    Parameters
    ----------
    tsv : numpy.ndarray
        The tsv label in the array format
    hop_size: int
        Hop size for the spectrogram. It will be used to convert tsvs into piano rolls
    max_midi: int
        The top bin of the pianoroll. Default 108 which corresponds to C8, the highest key on a standard 88-key piano
    min_midi: int
        The lowest bin of the pianoroll. Default 21 which corresponds to A0, the lowest key on a standard 88-key piano
    """
    
    
    n_keys = max_midi - min_midi + 1 # Calutate number of bins for the piano roll
    n_steps = (audio_length - 1) // hop_size + 1 # Calulate number of timesteps for the piano roll
    
    pianoroll = torch.zeros((n_steps, n_keys), dtype=int)
    velocity_roll = torch.zeros((n_steps, n_keys), dtype=int)
    
    for onset, offset, note, vel in tsv:
        left = int(round(onset * sample_rate / hop_size)) # Convert time to time step
        onset_right = min(n_steps, left + 1) # Ensure the time step of onset would not exceed the last time step
        frame_right = int(round(offset * sample_rate / hop_size))
        frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
        offset_right = min(n_steps, frame_right + 1)

        f = int(note) - min_midi
        pianoroll[left:onset_right, f] = 3 # assigning onset
        pianoroll[onset_right:frame_right, f] = 2 # assigning sustain
        pianoroll[frame_right:offset_right, f] = 1 # assigning offset
        velocity_roll[left:frame_right, f] = vel    
        
        
        
    return pianoroll, velocity_roll




def check_md5(path, md5_hash):
    with open(path, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()

        assert md5_returned==md5_hash, f"{os.path.basename(path)} is corrupted, please download it again"