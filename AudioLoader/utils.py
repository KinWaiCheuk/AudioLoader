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



# def get_segment(data, hop_size, sequence_length=None, max_midi=108, min_midi=21):
#     result = dict(path=data['path'])
#     audio_length = len(data['audio'])
#     pianoroll = data['pianoroll']
#     velocity_roll = data['velocity_roll']
# #     start = time.time()
# #     pianoroll, velocity_roll = tsv2roll(data['tsv'], audio_length, data['sr'], hop_size, max_midi, min_midi)
# #     print(f'tsv2roll time used = {time.time()-start}')
    
#     if sequence_length is not None:
#         # slicing audio
#         begin = np.random.randint(audio_length - sequence_length)
# #         begin = 1000 # for debugging
#         end = begin + sequence_length
#         result['audio'] = data['audio'][begin:end]
        
#         # slicing pianoroll
#         step_begin = begin // hop_size
#         n_steps = sequence_length // hop_size
#         step_end = step_begin + n_steps
#         labels = pianoroll[step_begin:step_end, :]
#         result['velocity'] = velocity_roll[step_begin:step_end, :]
#     else:
#         result['audio'] = data['audio']
#         labels = pianoroll
#         result['velocity'] = velocity_roll

# #     result['audio'] = result['audio'].float().div_(32768.0) # converting to float by dividing it by 2^15
#     result['onset'] = (labels == 3)
#     result['offset'] = (labels == 1)
#     result['frame'] = (labels > 1)
#     result['velocity'] = result['velocity']
#     # print(f"result['audio'].shape = {result['audio'].shape}")
#     # print(f"result['label'].shape = {result['label'].shape}")
#     return result


# def collect_batch(batch, hop_size, sequence_length, max_midi=108, min_midi=21):
#     frame = []
#     onset = []
#     offset = []
#     velocity = []
#     audio = torch.empty(len(batch), sequence_length)
#     path = []
    
#     # cut the audio into same sequence length and collect them
#     for idx, sample in enumerate(batch):
#         start = time.time()
#         results = get_segment(sample, hop_size, sequence_length, max_midi, min_midi)
# #         print(f'get_segment time used = {time.time()-start}')        
#         frame.append(results['frame'])
#         onset.append(results['onset'])
#         offset.append(results['offset'])
#         velocity.append(results['velocity'])
#         audio[idx] = results['audio']
#         path.append(results['path'])
        

#     output_batch = {'audio': audio,
#                     'frame': torch.tensor(frame).float(),
#                     'onset': torch.tensor(onset).float(), 
#                     'offset': torch.tensor(offset).float(),
#                     'velocity': torch.tensor(velocity).float(),
#                     'path': path
#                      }
    
#     return output_batch    