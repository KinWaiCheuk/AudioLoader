import torch
from torch import nn as nn, optim as optim
from torchaudio.functional import resample
import torch.distributions as tdist
from torch.utils.data import Dataset
import torchaudio


import os
from typing import Optional
import numpy as np
import pathlib
import pickle
import tqdm

from .utils import TargetProcessor
# from End2End.utils import int16_to_float32

from time import time

idx2occurrence_map= { # This map would be useful for weighted sampling
    0: 799.0,
    1: 630.0,
    2: 53.0,
    3: 74.0,
    4: 160.0,
    5: 237.0,
    6: 28.0,
    7: 50.0,
    8: 674.0,
    9: 969.0,
    10: 1320.0,
    11: 31.0,
    12: 5.0,
    13: 17.0,
    14: 5.0,
    15: 788.0,
    16: 14.0,
    17: 16.0,
    18: 397.0,
    19: 62.0,
    20: 59.0,
    21: 12.0,
    22: 42.0,
    23: 148.0,
    24: 226.0,
    25: 41.0,
    26: 5.0,
    27: 39.0,
    28: 13.0,
    29: 132.0,
    30: 20.0,
    31: 119.0,
    32: 207.0,
    33: 306.0,
    34: 1.0,
    35: 1.0,
    36: 1.0,
    37: 1.0,
    38: 1320.0,
    39: 0.0
}


class Slakh2100(Dataset):
    def __init__(
        self,
        split: str,
        waveform_dir: str,
        notes_pkls_dir,
        segment_seconds: str,
        frames_per_second: int,
        transcription: bool,
        random_crop: bool,
        source: bool,
        MIDI_MAPPING,
        sample_rate=16000,   
    ):
        r"""Instrument classification dataset takes the meta of an audio
        segment as input, and return the waveform, onset_roll, and targets of
        the audio segment. Dataset is used by DataLoader.

        Args:
            waveform_dir: str
            midi_events_hdf5s_dir: str
            segment_seconds: float, e.g., 2.0
            frames_per_second: int, e.g., 100
            augmentor: Augmentor
        """
        self.waveform_dir = waveform_dir
        self.notes_pkls_dir = notes_pkls_dir
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.transcription = transcription
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.source = source
        
        if self.segment_seconds:
            self.segment_samples = int(self.sample_rate * self.segment_seconds)
        else:
            self.segment_samples = None
            

        self.name_to_ix = MIDI_MAPPING.NAME_TO_IX
        self.ix_to_name = MIDI_MAPPING.IX_TO_NAME
        self.plugin_labels_num = MIDI_MAPPING.plugin_labels_num        

        
        # random seed
        # When using num_workers>0, this line will cause the same crop over and over again
#         self.random_state = np.random.RandomState(1234)

        # finding all audio tracks e.g. Track00001, ..., Track02100
        # These track names are used as the key to load the mix, and the sources
        flac_names = os.listdir(os.path.join(waveform_dir, split))    
        flac_names = sorted(flac_names)

        self.audio_name_list = []

        for flac_name in tqdm.tqdm(flac_names, desc=f'Loading {split} set track names'):
            self.audio_name_list.append([split, flac_name])        

        notes_pkls_dir = os.path.join(notes_pkls_dir, split)
        pkl_names = sorted(os.listdir(notes_pkls_dir))
        self.pkl_paths = [os.path.join(notes_pkls_dir, pkl_name) for pkl_name in pkl_names]

        self.total_dict = {}

        for audio_path in tqdm.tqdm(self.pkl_paths,desc=f'Loading {split} pkl files'):
            event_lists = pickle.load(open(audio_path, 'rb'))        
            self.total_dict[pathlib.Path(audio_path).stem] = event_lists
            

        self.target_processor = TargetProcessor(frames_per_second=frames_per_second,
                                                begin_note=21,
                                                classes_num=88
        )
        
    def __len__(self):
        return len(self.audio_name_list)

    def __getitem__(self, idx):
        r"""Get input and target of a segment for training.

        Args:
            meta: list, [split, hdf5_name, start_time], e.g.,
            ['train', 'Track00255.h5', 4.0]

        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num),
            'offset_roll': (frames_num, classes_num),
            'reg_onset_roll': (frames_num, classes_num),
            'reg_offset_roll': (frames_num, classes_num),
            'frame_roll': (frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num),
            'mask_roll':  (frames_num, classes_num),
            'pedal_onset_roll': (frames_num,),
            'pedal_offset_roll': (frames_num,),
            'reg_pedal_onset_roll': (frames_num,),
            'reg_pedal_offset_roll': (frames_num,),
            'pedal_frame_roll': (frames_num,)}
        """        
        split, flac_name = self.audio_name_list[idx]
        flac_path = os.path.join(self.waveform_dir,
                                  split,
                                  flac_name,
                                  'waveform.flac') # waveform.flac is the downsampled version of mix.flac
        
        
        waveform, sr = torchaudio.load(flac_path) # waveform shape (1, len)
        waveform =  waveform[0] # removing the first dimension
        
        assert sr==self.sample_rate,\
        f"{flac_path} is having a sampling rate of {sr}, which is different from the {self.sample_rate} that you choose."
        f"Please double check your audio files."

        data_dict = {}

        # Load segment waveform.
        audio_length = len(waveform)
        if self.segment_samples:
            assert (audio_length - self.segment_samples)>0, \
            f"sequence_length={self.segment_samples} is longer than the "
            f"audio_length={audio_length}. Please reduce the sequence_length"
            if self.random_crop:
                start_sample = np.random.randint(audio_length - self.segment_samples)
            else:
                start_sample = self.sample_rate*10
            start_time = start_sample/self.sample_rate
            end_sample = start_sample + self.segment_samples

            waveform_seg = waveform[start_sample : end_sample]            
            unique_plugin_names=[] # dummy value to start the following code
            while waveform_seg.sum()==0 or len(unique_plugin_names) == 0: # resample if the audio is empty                
    #             if waveform_seg.sum()==0:
    #                 print(f'{hdf5_name} waveform is empty')
    #             elif unique_plugin_names==0:
    #                 print(f'{hdf5_name} waveform not empty, but no instrumet ')
                if self.random_crop:
                    start_sample = np.random.randint(audio_length - self.segment_samples)
                else:
                    start_sample = start_sample + self.sample_rate # Shift the audio by 1 second if the audio is empty
                start_time = start_sample/self.sample_rate
                end_sample = start_sample + self.segment_samples
                
                
                waveform_seg = waveform[start_sample : end_sample]

                valid_length = len(waveform_seg)
                # (segment_samples,), e.g., (160000,)


                # pkl_path = os.path.join(self.notes_pkls_dir, '{}.pkl'.format(pathlib.Path(hdf5_name).stem))
                # events_dict = pickle.load(open(pkl_path, 'rb'))
                print(f"{flac_name=}")
                print(f"{pathlib.Path(flac_name).stem=}")
                event_list = self.total_dict[pathlib.Path(flac_name).stem] # why do I need .Path.stem?

                segment_notes_dict = {}

                for event in event_list:
                    if start_time < event['start'] < start_time + self.segment_seconds or start_time < event['end'] < start_time + self.segment_seconds:
                        plugin_name = event['plugin_name']
                        if plugin_name in segment_notes_dict.keys():
                            segment_notes_dict[plugin_name].append(event)
                        else:
                            segment_notes_dict[plugin_name] = [event]


                unique_plugin_names = list(segment_notes_dict.keys())                
            
        else:
            start_sample = 0
            start_time = 0
            end_sample = audio_length

            waveform_seg = waveform

            valid_length = len(waveform_seg)
            self.segment_seconds = valid_length/self.sample_rate
            # (segment_samples,), e.g., (160000,)

            # pkl_path = os.path.join(self.notes_pkls_dir, '{}.pkl'.format(pathlib.Path(hdf5_name).stem))
            # events_dict = pickle.load(open(pkl_path, 'rb'))
            event_list = self.total_dict[pathlib.Path(flac_name).stem]

            segment_notes_dict = {}

            for event in event_list:
                plugin_name = event['plugin_name']
                if plugin_name in segment_notes_dict.keys():
                    segment_notes_dict[plugin_name].append(event)
                else:
                    segment_notes_dict[plugin_name] = [event]

            unique_plugin_names = list(segment_notes_dict.keys())            
            

        if self.transcription:
            target_dict = {}
            if len(unique_plugin_names) == 0:
                # plugin_name = self.random_state.choice(PLUGIN_LABELS, size=1)[0]
                plugin_name = None
                prettymidi_events = []
            else:  
                for plugin_name in unique_plugin_names:
                    prettymidi_events = segment_notes_dict[plugin_name]
                    target_dict_per_plugin, note_events = self.target_processor.process2(start_time=start_time, 
                                                                                         segment_seconds=self.segment_seconds,
                                                                                         prettymidi_events=prettymidi_events,
                                                                                        )
                    target_dict[plugin_name] = target_dict_per_plugin               
            data_dict['target_dict'] = target_dict
            
        if self.source:
            source_dict = {}
            source_mask = {}
            split, flac_name = self.audio_name_list[idx]
#             with h5py.File(hdf5_path, 'r') as hf:
#                 for plugin_name in unique_plugin_names:
#                     try:
#                         source_dict[plugin_name] = int16_to_float32(hf['sources'][plugin_name][()][start_sample : end_sample])
#                         source_mask[plugin_name] = True
#                     except:
#                         source_dict[plugin_name] = np.zeros(end_sample-start_sample, dtype=np.float32)
#                         source_mask[plugin_name] = False        
            for plugin_name in unique_plugin_names:
                source_path = os.path.join(self.waveform_dir,
                                           split,
                                           flac_name,
                                           f'{plugin_name}.flac')
#                
                              
                try:
                    print(f"{source_path=}")
                    source_waveform, sr = torchaudio.load(source_path)
                    source_waveform = source_waveform[0,start_sample : end_sample]
                    assert sr==self.sample_rate,\
                    f"{source_path} is having a sampling rate of {sr}, which is different from the {self.sample_rate} that you choose."
                    f"Please double check your audio files."
                    source_dict[plugin_name] = source_waveform
                    source_mask[plugin_name] = True
                except Exception as e:
                    print(e)
                    source_dict[plugin_name] = np.zeros(end_sample-start_sample, dtype=np.float32)
                    source_mask[plugin_name] = False        
            data_dict['sources'] = source_dict
            data_dict['source_masks'] = source_mask
        
        data_dict['waveform'] = waveform_seg
        data_dict['start_sample'] = start_sample
        data_dict['valid_length'] = valid_length
        data_dict['flac_name'] = flac_name        

        target = np.zeros(self.plugin_labels_num)  # (plugin_names_num,)
        plugin_ids = [self.name_to_ix[plugin_name] for plugin_name in unique_plugin_names]
        for plugin_id in plugin_ids:          
            target[plugin_id] = 1
        data_dict['instruments'] = target
        
        if plugin_name:
            data_dict['plugin_id'] = self.name_to_ix[plugin_name]
        else:
            data_dict['plugin_id'] = None


        return data_dict  


class End2EndBatchDataPreprocessor:
    """
    Make this unwrapping a batch (BatchUnwrapping)
    For example, in a batch with 1 waveforms and 3 conditions,
    make it a batch size 3 with the following 3 samples
    (wave1, cond1, roll1), (wave1, cond2, roll2), (wave1, cond3, roll3).
    
    By doing so, the feedforward
    """
    
    def __init__(self,
                 MIDI_MAPPING,
                 mode,
                 temp,
                 samples,
                 neg_samples,
                 transcription,
                 source_separation,
                 audio_noise=None):               
        self.device = 'cuda'
        self.random_state = np.random.RandomState(1234)
        if mode=='full':
            self.process = self._sample_full
        elif mode=='random':
            self.process = self._sample_random
        elif mode=='imbalance':
            self.process = self._sample_imbalance
            
        self.name_to_ix = MIDI_MAPPING.NAME_TO_IX
        self.ix_to_name = MIDI_MAPPING.IX_TO_NAME
        self.plugin_labels_num = MIDI_MAPPING.plugin_labels_num
        self.temp = temp
        self.inst_samples = samples
        self.neg_inst_samples = neg_samples
        self.total_samples = self.inst_samples+self.neg_inst_samples # This is for mining both pos and neg training samples
        
        self.transcription = transcription
        self.source_separation = source_separation
        
        if audio_noise:
            self.noise = tdist.Normal(0,audio_noise)
        else:
            self.noise = None
            
    def __call__(self, batch):
        return self.process(batch)
    
    def _sample_full(self, batch):
        # Extracting conditions and unwrapping batch
        M = batch['instruments'].sum()
        instruments_per_wav = batch['instruments'].sum(1)
        counter = 0
        unwrapped_batch = {}       
        for n in range(len(batch['instruments'])): # Looping through the batchsize
            
            plugin_ids = torch.where(batch['instruments'][n]==1)[0]
            unwrapped_batch[n] = {}
            num_instruments = int(instruments_per_wav[n]) # use it as the new batch size

            # Placeholders for a new batch
            waveforms = torch.zeros(num_instruments, batch['waveform'].size(-1))
            conditions = torch.zeros(num_instruments, self.plugin_labels_num)
            # TODO: replace this hard encoded T=1001 with something better
            
            # creating placeholders for target_dict
            target_dict = {}
            for roll_type in batch['target_dict'][n][self.ix_to_name[int(plugin_ids[0])]].items():
                target_dict[roll_type[0]] = torch.zeros((num_instruments, *roll_type[1].shape)).to(self.device)
            for idx, plugin_id in enumerate(plugin_ids):
                conditions[idx, plugin_id] = 1
                waveforms[idx] = batch['waveform'][n]
                for roll_type in batch['target_dict'][n][self.ix_to_name[int(plugin_id)]].items():
                    target_dict[roll_type[0]][idx] = torch.from_numpy(roll_type[1])

        unwrapped_batch[n]['waveforms'] = waveforms.to(self.device) 
        unwrapped_batch[n]['conditions'] = conditions.to(self.device)
        unwrapped_batch[n]['target_dict'] = target_dict
            
        #['waveform', 'valid_length', 'target_dict', 'instruments', 'plugin_id'])
        return unwrapped_batch
    
    def _sample_random(self, batch):
        # Extracting basic information
        M = batch['instruments'].sum()
        batch_size = len(batch['instruments'])

        # create placeholders for target_dict
        # get a random key
        
        if self.transcription:
            key = list(batch['target_dict'][0].keys())[0]         
            target_dict = {}

            for roll_type in batch['target_dict'][0][key].items():
                target_dict[roll_type[0]] = torch.zeros((batch_size*self.total_samples, *roll_type[1].shape)).to(self.device)

        # Placeholders for a new batch
        
        waveforms = torch.zeros(batch_size*self.total_samples, batch['waveform'].size(-1))
        sources = torch.zeros(batch_size*self.total_samples, batch['waveform'].size(-1))
        masks = torch.zeros(batch_size*self.total_samples, 1)
        conditions = torch.zeros(batch_size*self.total_samples, self.plugin_labels_num)

        unwrapped_batch = {}       
        for n in range(batch_size): # Looping through the batchsize
            plugin_ids = torch.where(batch['instruments'][n]==1)[0].cpu()
            neg_plugin_ids = torch.where(batch['instruments'][n]==0)[0].cpu()
            
            # random sampling
            plugin_ids = np.random.choice(plugin_ids, self.inst_samples)
            neg_plugin_ids = np.random.choice(neg_plugin_ids, self.neg_inst_samples)
            
            # packing different instruenmts into the new batch
            for idx, plugin_id in enumerate(plugin_ids):
                conditions[n*self.total_samples+idx, plugin_id] = 1
                waveforms[n*self.total_samples+idx] = batch['waveform'][n] # repeat the same waveform for different instruments
                if 'sources' in batch.keys():
                    sources[n*self.total_samples+idx] = torch.from_numpy(batch['sources'][n][self.ix_to_name[int(plugin_id)]])
                    masks[n*self.total_samples+idx] = batch['source_masks'][n][self.ix_to_name[int(plugin_id)]]
                if self.noise:
                    waveforms[n*self.total_samples+idx] += self.noise.sample(batch['waveform'][n].shape) # adding noise to waveform
                if self.transcription:                    
                    for roll_type in batch['target_dict'][n][self.ix_to_name[int(plugin_id)]].items():
                        target_dict[roll_type[0]][n*self.total_samples+idx] = torch.from_numpy(roll_type[1])
                    
                    
            # packing negative instruenmts into the new batch
            for neg_idx, neg_plugin_id in enumerate(neg_plugin_ids):
                conditions[n*self.total_samples+idx+neg_idx+1, neg_plugin_id] = 1
                waveforms[n*self.total_samples+idx+neg_idx+1] = batch['waveform'][n] # repeat the same waveform for different instruments
                if 'sources' in batch.keys():                
                    sources[n*self.total_samples+idx+neg_idx+1] = torch.zeros_like(batch['waveform'][n]) # create an empty waveform for neg sample
                    masks[n*self.total_samples+idx+neg_idx+1] = batch['source_masks'][n][self.ix_to_name[int(plugin_id)]]               
                if self.noise:
                    waveforms[n*self.total_samples+idx+1] += self.noise.sample(batch['waveform'][n].shape) # adding noise to waveform
                if self.transcription:                    
                    for roll_type in batch['target_dict'][n][self.ix_to_name[int(plugin_id)]].items():
                        target_dict[roll_type[0]][n*self.total_samples+idx+neg_idx+1] = torch.zeros_like(torch.from_numpy(roll_type[1]))
                    
                    


        unwrapped_batch['waveforms'] = waveforms.to(self.device) 
        unwrapped_batch['conditions'] = conditions.to(self.device)
        if self.transcription:        
            unwrapped_batch['target_dict'] = target_dict
        unwrapped_batch['sources'] = sources.to(self.device)
        unwrapped_batch['source_masks'] = masks.to(self.device)
        #['waveform', 'valid_length', 'target_dict', 'instruments', 'plugin_id'])
        return unwrapped_batch  
    
    def _sample_imbalance(self, batch):
        # Extracting basic information
        M = batch['instruments'].sum()
        batch_size = len(batch['instruments'])

        # create placeholders for target_dict
        # get a random key
        
        if self.transcription:
            key = list(batch['target_dict'][0].keys())[0]         
            target_dict = {}

            for roll_type in batch['target_dict'][0][key].items():
                target_dict[roll_type[0]] = torch.zeros((batch_size*self.total_samples, *roll_type[1].shape)).to(self.device)

        # Placeholders for a new batch
        
        waveforms = torch.zeros(batch_size*self.total_samples, batch['waveform'].size(-1))
        sources = torch.zeros(batch_size*self.total_samples, batch['waveform'].size(-1))
        masks = torch.zeros(batch_size*self.total_samples, 1)
        conditions = torch.zeros(batch_size*self.total_samples, self.plugin_labels_num)

        unwrapped_batch = {}       
        for n in range(batch_size): # Looping through the batchsize
            plugin_ids = torch.where(batch['instruments'][n]==1)[0].cpu()
            neg_plugin_ids = torch.where(batch['instruments'][n]==0)[0].cpu()
            
            # sampling uncommon instrument more often
            occurrence = np.array(list(map(idx2occurrence_map.get, plugin_ids.cpu().tolist())))
            temp_occ = (1/occurrence)**self.temp
            inverse_prob = temp_occ/temp_occ.sum()
            plugin_ids = np.random.choice(plugin_ids, self.inst_samples, p=inverse_prob)
            neg_plugin_ids = np.random.choice(neg_plugin_ids, self.neg_inst_samples)
            
            # packing different instruenmts into the new batch
            for idx, plugin_id in enumerate(plugin_ids):
                conditions[n*self.total_samples+idx, plugin_id] = 1
                waveforms[n*self.total_samples+idx] = batch['waveform'][n] # repeat the same waveform for different instruments
                if 'sources' in batch.keys():
                    sources[n*self.total_samples+idx] = torch.from_numpy(batch['sources'][n][self.ix_to_name[int(plugin_id)]])
                    masks[n*self.total_samples+idx] = batch['source_masks'][n][self.ix_to_name[int(plugin_id)]]
                if self.noise:
                    waveforms[n*self.total_samples+idx] += self.noise.sample(batch['waveform'][n].shape) # adding noise to waveform
                if self.transcription:                    
                    for roll_type in batch['target_dict'][n][self.ix_to_name[int(plugin_id)]].items():
                        target_dict[roll_type[0]][n*self.total_samples+idx] = torch.from_numpy(roll_type[1])
                    
                    
            # packing negative instruenmts into the new batch
            for neg_idx, neg_plugin_id in enumerate(neg_plugin_ids):
                conditions[n*self.total_samples+idx+neg_idx+1, neg_plugin_id] = 1
                waveforms[n*self.total_samples+idx+neg_idx+1] = batch['waveform'][n] # repeat the same waveform for different instruments
                if 'sources' in batch.keys():                
                    sources[n*self.total_samples+idx+neg_idx+1] = torch.zeros_like(batch['waveform'][n]) # create an empty waveform for neg sample
                    masks[n*self.total_samples+idx+neg_idx+1] = batch['source_masks'][n][self.ix_to_name[int(plugin_id)]]               
                if self.noise:
                    waveforms[n*self.total_samples+idx+1] += self.noise.sample(batch['waveform'][n].shape) # adding noise to waveform
                if self.transcription:                    
                    for roll_type in batch['target_dict'][n][self.ix_to_name[int(plugin_id)]].items():
                        target_dict[roll_type[0]][n*self.total_samples+idx+neg_idx+1] = torch.zeros_like(torch.from_numpy(roll_type[1]))
                    
                    


        unwrapped_batch['waveforms'] = waveforms.to(self.device) 
        unwrapped_batch['conditions'] = conditions.to(self.device)
        if self.transcription:        
            unwrapped_batch['target_dict'] = target_dict
        unwrapped_batch['sources'] = sources.to(self.device)
        unwrapped_batch['source_masks'] = masks.to(self.device)
        #['waveform', 'valid_length', 'target_dict', 'instruments', 'plugin_id'])
        return unwrapped_batch    
        
    
class FullPreprocessor:
    """
    Make this unwrapping a batch (BatchUnwrapping)
    For example, in a batch with 1 waveforms and 3 conditions,
    make it a batch size 3 with the following 3 samples
    (wave1, cond1, roll1), (wave1, cond2, roll2), (wave1, cond3, roll3).
    
    By doing so, the feedforward
    """
    
    def __init__(self, MIDI_MAPPING):               
        self.random_state = np.random.RandomState(1234)
        self.process = self._sample_full

        self.name_to_ix = MIDI_MAPPING.NAME_TO_IX
        self.ix_to_name = MIDI_MAPPING.IX_TO_NAME
        self.plugin_labels_num = MIDI_MAPPING.plugin_labels_num

            
    def __call__(self, batch):
        return self.process(batch)
    
    def _sample_full(self, batch):
        # Extracting conditions and unwrapping batch
        batch_size = len(batch['instruments'])
        num_classes = batch['instruments'].shape[-1]
        counter = 0
        unwrapped_batch = {}       
        
        # randomly pick one inst to get the pianoroll shape information
        inst_str = list(batch['target_dict'][0].keys())[0]
        roll_shape = batch['target_dict'][0][inst_str]['onset_roll'].shape
        # creating placeholders for target roll
        target_roll = torch.zeros(batch_size, num_classes, *roll_shape).to(batch['instruments'].device)            
        
        plugin_id_list = []
        for n in range(batch_size): # Looping through the batchsize
            plugin_ids = torch.where(batch['instruments'][n]==1)[0]
            plugin_id_list.append(plugin_ids)
            for ix in plugin_ids:
                target_roll[n, ix] = torch.from_numpy(batch['target_dict'][n][self.ix_to_name[ix.item()]]['frame_roll'])
            
        #['waveform', 'valid_length', 'target_dict', 'instruments', 'plugin_id']                                                         
        unwrapped_batch['waveforms'] = batch['waveform']
        unwrapped_batch['plugin_ids'] = plugin_id_list
        unwrapped_batch['target_roll'] = target_roll   
        return unwrapped_batch
     
        


def collate_slakh(list_data_dict):
    r"""Collate input and target of segments to a mini-batch.

    Args:
        list_data_dict: e.g. [
            {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...},
            {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...},
            ...]

    Returns:
        data_dict: e.g. {
            'waveform': (batch_size, segment_samples)
            'frame_roll': (batch_size, segment_frames, classes_num),
            ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        if key in ['plugin_id', 'segment_notes_dict', 'target_dict', 'sources', 'source_masks']:
            np_data_dict[key] = [data_dict[key] for data_dict in list_data_dict]
        elif key in ['list_at_onset_rolls', 'list_at_segments']:
            np_data_dict[key] = [torch.Tensor(data_dict[key]) for data_dict in list_data_dict]
        elif key=='hdf5_name':
            np_data_dict[key] = [data_dict[key] for data_dict in list_data_dict]
        else:
            np_data_dict[key] = torch.Tensor(np.array([data_dict[key] for data_dict in list_data_dict]))

    return np_data_dict