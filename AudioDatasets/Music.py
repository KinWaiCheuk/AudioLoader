import json
import os
from abc import abstractmethod
from glob import glob
import sys
import pickle
import numpy as np
# import soundfile
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.compliance import kaldi # for downsampling
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

"""
This file is based on https://github.com/jongwook/onsets-and-frames

TODA:
1. Don't load everything on RAM, load only the files when needed (or only do it with MAESTRO?)
2. 
"""

# Helper functions for midi to tsv conversions
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


def process(input_file, output_file):
    midi_data = parse_midi(input_file)
    np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')


def files(file_list, output_dir=False):
    for input_file in tqdm(file_list, desc='Converting midi to tsv:'):
        if input_file.endswith('.mid'):
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
        


class PianoRollAudioDataset(Dataset):
    def __init__(self,
                 groups=None,
                 sequence_length=None,
                 seed=42,
                 refresh=False,
                 device='cpu',
                 download=False):
        
        
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed) # for spectrogram segmentation
        self.refresh = refresh
        self.data = []
        
        if download:
            if os.path.isdir(self.root):
                print(f'Dataset folder exists, skipping download...')
                decision = input(f"Do you want to extract {self.name_archive+self.ext_archive} again?\n"+
                                 f"This action will overwrite exsiting files, do you still want to continue? [yes/no]")                
            else:
                decision='yes' 
                if not os.path.isdir(self.root):
                    print(f'Creating download path = {self.root}')
                    os.makedirs(os.path.join(self.root))
    #                 if os.path.isfile(download_path+ext_archive):
    #                     print(f'.tar.gz file exists, proceed to extraction...')
    #                 else:
                if os.path.isfile(os.path.join(self.root, self.name_archive+self.ext_archive)):
                    print(f'{download_path+ext_archive} already exists, proceed to extraction...')
                else:
                    print(f'downloading...')
                    download_url(self.url, root, hash_value=self.checksum, hash_type='md5')

            if decision.lower()=='yes':
                print(f'Extracting main folder...')
                extract_archive(os.path.join(self.root, self.name_archive+self.ext_archive))
                for group in groups:
                    group_path = os.path.join(path, i)
                    if not os.path.isdir():
                        print(f'Extracting sub-folder {group}...')
                        extract_archive(os.path.join(self.root, self.name_archive+'zip'))
                
        else:
            if not os.path.isdir(path):
                raise ValueError(f'{path} not found, please specify the correct path')  
                
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group): #self.files is defined in MAPS class
                self.data.append(self.load(*input_files)) # self.load is a function defined below. It first loads all data into memory first
    def __getitem__(self, index):

        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            # print(f'step_begin = {step_begin}')
            
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
#             result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
#             result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
#         result['velocity'] = result['velocity'].float().div_(128.0)
        # print(f"result['audio'].shape = {result['audio'].shape}")
        # print(f"result['label'].shape = {result['label'].shape}")
        return result

    def __len__(self):
        return len(self.data)

    @classmethod # This one seems optional?
    @abstractmethod # This is to make sure other subclasses also contain this method
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels
        Returns
        -------
            A dictionary containing the following data:
            path: str
                the path to the audio file
            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform
            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path) and self.refresh==False: # Check if .pt files exist, if so just load the files
            return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio) # convert numpy array to pytorch tensor
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1 # This will affect the labels time steps

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
            onset_right = min(n_steps, left + HOPS_IN_ONSET) # Ensure the time step of onset would not exceed the last time step
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data
    
    
class MAPS(Dataset):
    def __init__(self,
                 root='./MAPS',
                 groups='all',
                 data_type='MUS',
                 overlap=True,
                 refresh=False,
                 download=False,
                 ext_audio='.wav'):
        """
        root (str): The folder that contains the MAPS dataset folder
        groups (list or str): Choose which sub-folders to load. Avaliable choices are 
                              `train`, `test`, `all`. Default is `all`, which means loading
                               all sub-folders. Alternatively, users can provide a list of
                               subfolders to be loaded.  
        data_type (str): Four different types of data are available, `MUS`, `ISOL`, `RAND`, `UCHO`.
                         `MUS` is the default setting which stands for full music pieces .
                         
        """
        
        self.overlap = overlap
        super().__init__()
        
        self.url = "https://amubox.univ-amu.fr/s/iNG0xc5Td1Nv4rR/download"
        self.checksum = '02a8f140dc9a7c85639b0c01e5522add'
        self.root = root
        self.ext_archive = '.tar'
        self.name_archive = 'MAPS'
        self.data_type = data_type
        self.ext_audio = ext_audio
        self.refresh = refresh
             
        groups = groups if isinstance(groups, list) else self.available_groups(groups)
        self.groups = groups

        if download:
            if os.path.isdir(os.path.join(self.root, self.name_archive)):
                print(f'Dataset folder exists, skipping download...\n'
                      f'Checking sub-folders...')
                self.extract_subfolders(groups)
                self.extract_tsv()
            elif os.path.isfile(os.path.join(self.root, self.name_archive+self.ext_archive)):
                print(f'.tar file exists, skipping download...')
                print(f'Extracting MAPS.tar')
                extract_archive(os.path.join(self.root, self.name_archive+self.ext_archive))
                self.extract_subfolders(groups)
                self.extract_tsv()                
            else:
                if not os.path.isdir(self.root):
                    print(f'Creating download path = {self.root}')
                    os.makedirs(os.path.join(self.root))
                    
                print(f'Downloading from {self.url}...')
                download_url(self.url, root, hash_value=self.checksum, hash_type='md5')
                print(f'Extracting MAPS.tar')
                extract_archive(os.path.join(self.root, self.name_archive+self.ext_archive))
                self.extract_subfolders(groups)
                self.extract_tsv()
        
        else:
            if os.path.isdir(os.path.join(root, self.name_archive)):
                print(f'MAPS folder found, checking content integrity...')
                self.extract_subfolders(groups)   
            else:
                raise ValueError(f'{root} does not contain the MAPS folder, '
                                 f'please specify the correct path or download it by setting `download=True`')  
                
#         print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
#               f"of {self.__class__.__name__} at {os.path.join(self.root, self.name_archive)}")
        self._walker = []
        for group in groups:
            wav_paths = glob(os.path.join(self.root, self.name_archive, group, data_type, f'*{ext_audio}'))
            self._walker.extend(wav_paths)
                
        print(f'{len(self._walker)} audio files found')

    def extract_subfolders(self, groups):
        for group in groups:
            group_path = os.path.join(self.root, self.name_archive, group)
            if not os.path.isdir(group_path):
                print(f'Extracting sub-folder {group}...', end='\r')
                if group=='ENSTDkAm':
                    # ENSTDkAm consists of ENSTDkAm1.zip and ENSTDkAm2.zip
                    # Extract and merge both ENSTDkAm1.zip and ENSTDkAm2.zip as ENSTDkAm
                    extract_archive(os.path.join(self.root, self.name_archive, group+'1.zip'))
                    extract_archive(os.path.join(self.root, self.name_archive, group+'2.zip'))
                else:
                    extract_archive(os.path.join(self.root, self.name_archive, group+'.zip'))
                print(f' '*50, end='\r')
                print(f'{group} extracted.')
                    
    def extract_tsv(self):
        """
        Convert midi files into tsv files for easy loading.
        """
        
        tsvs = glob(os.path.join(self.root, self.name_archive, '*', self.data_type, '*.tsv'))
        num_tsvs = len(tsvs)
        if num_tsvs>0:
            decision = input(f"There are already {num_tsvs} tsv files.\n"+
                             f"Do you want to overwrite them? [yes/no]")
        elif num_tsvs==0:
            decision='yes'
            
        if decision.lower()=='yes':
            midis = glob(os.path.join(self.root, self.name_archive, '*', self.data_type, '*.mid')) # loading lists of midi    
            Parallel(n_jobs=multiprocessing.cpu_count())\
                    (delayed(process)(in_file, out_file) for in_file, out_file in files(midis, output_dir=False))
                
    def available_groups(self, group):
        if group=='train':
            return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']
        elif group=='test':
            return ['ENSTDkAm', 'ENSTDkCl']
        elif group=='all':
            return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']
        
        
    def __getitem__(self, index):
        """
        load an audio track and the corresponding labels
        Returns
        -------
            A dictionary containing the following data:
            path: str
                the path to the audio file
            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform
            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else
            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        audio_path = self._walker[index]
        tsv_path = audio_path.replace(self.ext_audio, '.tsv')
        saved_data_path = audio_path.replace(self.ext_audio, '.pt')
        if os.path.exists(audio_path.replace(self.ext_audio, '.pt')) and self.refresh==False: 
            # Check if .pt files exist, if so just load the files
            return torch.load(saved_data_path)
        # Otherwise, create the .pt files
        waveform, sr = torchaudio.load(audio_path)
        if waveform.dim()==2:
            waveform = waveform.mean(0) # converting a stereo track into a mono track
        audio_length = len(waveform)

#         n_keys = MAX_MIDI - MIN_MIDI + 1
#         n_steps = (audio_length - 1) // HOP_LENGTH + 1 # This will affect the labels time steps

#         label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
#         velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

#         for onset, offset, note, vel in midi:
#             left = int(round(onset * SAMPLE_RATE / HOP_LENGTH)) # Convert time to time step
#             onset_right = min(n_steps, left + HOPS_IN_ONSET) # Ensure the time step of onset would not exceed the last time step
#             frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
#             frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
#             offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

#             f = int(note) - MIN_MIDI
#             assert f>0, f"Found midi note number {int(note)}, while MIN_MIDI={MIN_MIDI}. Please change your MIN_MIDI."
#             label[left:onset_right, f] = 3
#             label[onset_right:frame_right, f] = 2
#             label[frame_right:offset_right, f] = 1
#             velocity[left:frame_right, f] = vel

#         data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)\
        data = dict(path=audio_path, sr=sr, audio=waveform, tsv=tsv)
        torch.save(data, saved_data_path)
        return data        

#     def files(self, group, music_type):
#         return 
        
#         flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
#         if self.overlap==False:
#             with open('overlapping.pkl', 'rb') as f:
#                 test_names = pickle.load(f)
#             filtered_flacs = []    
#             for i in flacs:
#                 if any([substring in i for substring in test_names]):
#                     pass
#                 else:
#                     filtered_flacs.append(i)
#             flacs = filtered_flacs 
#         # tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]
#         tsvs = [f.replace('/flac/', '/tsvs/').replace('.flac', '.tsv') for f in flacs]
#         assert(all(os.path.isfile(flac) for flac in flacs))
#         assert(all(os.path.isfile(tsv) for tsv in tsvs))
        
#         print(f'len(flacs) = {len(flacs)}')
#         print(f'tsvs = {len(tsvs)}')

#         return sorted(zip(flacs, tsvs))    

    def __len__(self):
        return len(self._walker)
    
    
    def resample(self, sr, output_format='flac'):
        """
        ```python
        dataset.resample(sr, output_format='flac')
        dataset = MAPS('./Folder', groups='all', ext_audio='.flac')
        ```
        
        Resample audio clips to the target sample rate `sr` and the target format `output_format`.
        This method requires `pydub`.
        After resampling, you need to create another instance of `MAPS` in order to load the new
        audio files instead of the original `.wav` files.
        """
        
        from pydub import AudioSegment        
        def _resample(wavfile, sr, output_format):
            sound = AudioSegment.from_wav(wavfile)
            sound = sound.set_frame_rate(sr) # downsample it to sr
            sound = sound.set_channels(1) # Convert Stereo to Mono
            sound.export(wavfile[:-3] + output_format, format=output_format)            
            
        Parallel(n_jobs=multiprocessing.cpu_count())\
        (delayed(_resample)(wavfile, sr, output_format)\
         for wavfile in tqdm(self._walker,
                             desc=f'Resampling to {sr}Hz .{output_format} files'))

    
    def clear_audio(self, audio_format='.flac'):
        clear_list = []
        for group in self.groups:
            audio_paths = glob(os.path.join(self.root,
                                          self.name_archive,
                                          group,
                                          self.data_type,
                                          f'*{audio_format}'))
            clear_list.extend(audio_paths)
            
        num_files = len(clear_list)
        if num_files>0:
            decision = input(f"{num_files} files found, do you want to clear them?[yes/no]")
            if decision.lower()=='yes':
                for i in clear_list:
                    os.remove(i)
            elif decision.lower()=='no':
                print(f'aborting...')