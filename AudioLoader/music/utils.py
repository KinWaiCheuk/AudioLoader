import torch
import numpy as np
import time
from tqdm import tqdm
import hashlib
from mido import Message, MidiFile, MidiTrack
import csv

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

# Helper functions for midi to tsv conversions
def parse_csv(path):
    with open(path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None) # skipping the header
        notes = [] # container for storing tsv entries
        for row in csv_reader:
            # It is fixed to 44100 because the original sr
            onset = int(row[0])/44100 # converting samples to second
            offset = int(row[1])/44100
            pitch = int(row[3])
            velocity = 127
            note = (onset, offset, pitch, velocity)
            notes.append(note)

    return np.array(notes)

def parse_midi(path):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = MidiFile(path)

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
#             offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n is events[-1])
            offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n['note'] == offset['note'] or n is events[-1])
        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)

def process_csv(input_file, output_file):
    """Parsing CSV files from MusicNet"""
    csv_data = parse_csv(input_file)
    np.savetxt(output_file, csv_data, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')


def process_midi(input_file, output_file):
    midi_data = parse_midi(input_file)
    np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')


def files(file_list, ext='.mid', output_dir=False):
    for input_file in tqdm(file_list, desc='Converting midi to tsv:'):
        if input_file.endswith('.mid') or input_file.endswith('.csv') :
            if output_dir==False:
                output_file = input_file[:-4] + '.tsv'
            else:
                output_file = os.path.join(output_dir, os.path.basename(input_file[:-4]) + '.tsv')
        elif input_file.endswith('.midi'):
            if output_dir==False:
                output_file = input_file[:-5] + '.tsv'
            else:
                output_file = os.path.join(output_dir, os.path.basename(input_file[:-5]) + '.tsv')                
        else:
            print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
            continue

        yield (input_file, output_file)
        
    