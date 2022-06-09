# Automatic Music Transcription

## MAPS
### Introduction
[MAPS](https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR?path=%2F) dataset contains 9 folders, each folder contains 30 full music recordings and the aligned midi annoations. The two folders `ENSTDkAm` and `ENSTDkCl` contains real acoustic recording obtained from a YAMAHA Disklavier. The rest are synthesized audio clips. This ready-to-use PyTorch `Dataset` will automatically set up most of the things.

### Usage
To use this dataset for the first time, set `download=True`. 

```python
from AudioLoader.music.amt import MAPS
dataset = MAPS(root='./',
               groups='all',
               data_type='MUS',
               overlap=True,
               use_cache=True,
               download=True,
               preload=False,
               sequence_length=None,
               seed=42,
               hop_length=512,
               max_midi=108,
               min_midi=21,
               ext_audio='.wav',
               sampling_rate=None)
```

This will download, unzip, and extract the `.tsv` labels.

If `use_cache=True`, the output dictionary containing `(path, audio, velocity, onset, offset, frame, sr)` will be saved as a `.pt` file. Loading from `.pt` files is slightly faster.

If `preload=True`, the whole dataset will be loaded into RAM, which allows a faster data accessing for each iteration. If the dataset is too large to be loaded in RAM, you can set `preload=False`

If `sequence_length=None`, it will load the full audio track, otherwise, the output will be automatically crop to `sequence_length`. If `hop_length` is set to be same as the one used in spectrogram extraction, the piano rolls returned by this dataset will be aligned with the spectrogram. 

`groups` controls which folders in MAPS to be loaded. `groups='all'` loads all folders; `groups='train'` loads the train set; `groups=test` loads the test set. Alternatively, you can also pass a list of the folder name `['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD']` to load the folders you want.

`ext_audio`: If you have resampled your audio into `flac` format, you can use this to control which set of audio to load.

`sampling_rate`: Set it to the number you want to downsample to. The downsampled audio clips are in `.flac` format, so you need to set `ext_audio='.flac'` to load them.

`dataset[i]` returns a dictionary containing:

```python
{'path': './MusicDataset/MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.wav',
 'audio': tensor([-0.0123, -0.0132, -0.0136,  ...,  0.0002, -0.0026, -0.0051]),
 'velocity': tensor([[...]]),
 'onset': tensor([[...]]),
 'offset': tensor([[...]]),
 'frame': tensor([[...]]),
 'sr': 44100}
```

`frame`: piano rolls of the shape `[T,K]`, where `T` is the number of timesteps, `K` is the number of midi notes. 

`onset`: onset locations in piano roll form `[T, K]`.

`offset`: offset locations in piano roll form `[T, K]`.

`sr`: sampling rate of the audio clip
Each row of `midi` represents a midi note, and it contains the information: `[start_time, end_time, Midi_pitch, velocity]`.

The original audio clips are all steoro, this PyTorch dataset automatically convert them back to mono tracks. Alternatively, the `.resample()` method can be also used to resample and convert tracks back to mono.

### Getting a batch of audio segment
```python
loader = DataLoader(dataset, batch_size=4)
for batch in loader:
    audios = batch['audio'].to(device)
    frames = batch['frame'].to(device)
```

### Other functionalities

1. #### resample
```python
dataset.resample(sr, output_format='flac', num_threads=-1)
dataset = MAPS('./Folder', groups='all', ext_audio='.flac')
```
Resample audio clips to the target sample rate `sr` and the target format `output_format`. This method requires `pydub`. After resampling, you need to create another instance of `MAPS` in order to load the new audio files instead of the original `.wav` files.

`num_threads` sets the number of threads to use when resampling audio clips. The default value `-1` is to use all available threads. If corrupted audio clips were produced when using `num_threads>0`, set `num_threads=0` to completely disable multithreading.
 

2. #### extract_tsv
```python
dataset.extract_tsv()
```
Convert midi files into tsv files for easy loading.

3. #### clear_caches
```python
dataset.clear_caches()
```
Removing existing caches files.


## MusicNet
### Introduction
[MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html)
dataset contains 330 classical music recordings, such as
string quartet, horn piano trio, and solo flute.
The train set consists of 320 recordings, and the test set contains the remaining 10 recodings.
### Usage
To use this dataset for the first time, set `download=True`. 

```python
from AudioLoader.music.amt import MusicNet
dataset = MusicNet('./',
                    groups='all',
                    ext_audio='.flac',
                    use_cache=False,
                    download=False
                    preload=True,
                    sequence_length=sequence_length,
                    seed=42,
                    hop_length=512,
                    max_midi=108,
                    min_midi=21,
                    ext_audio='.wav'
                   )
```

This will download, unzip, and convert the `.csv` files into `.tsv` files.

All arugments are same as the [MAPS](#MAPS) dataset, you may want to check the avaliable arugments [here](#MAPS).

`dataset[i]` returns a dictionary containing:

```python
{'path': './MusicDataset/musicnet/train_data/1788.wav',
 'audio': tensor([ 0.0179,  0.0264,  0.0230,  ..., -0.1169, -0.0786, -0.0732]),
 'velocity': tensor([[...]]),
 'onset': tensor([[...]]),
 'offset': tensor([[...]]),
 'frame': tensor([[...]]),
 'sr': 44100}
```


### Getting a batch of audio segment
```python
loader = DataLoader(dataset, batch_size=4)
for batch in loader:
    audios = batch['audio'].to(device)
    frames = batch['frame'].to(device)
```

### Other functionalities

Same as [MAPS](./README.md#MAPS)


## MAESTRO
### Introduction
This repository is designed for [MAESTROV2.0.0.](https://magenta.tensorflow.org/datasets/maestro#v200). It contains about 200 hours of piano track. Metadata of each track is included in maestro-v2.0.0.csv and maestro-v2.0.0.json. The train set, validation set and test set consists of 967 tracks, 137 tracks and 178 tracks respectively.

### Usage
To use this dataset for the first time, set `download=True`. 
```python
from AudioLoader.music.amt import MAESTRO
dataset = MAESTRO(root='./',
               groups=['train'],                
               use_cache=True,
               download=True,
               preload=False,
               sequence_length=None,
               seed=42,
               hop_length=512,
               max_midi=108,
               min_midi=21,
               ext_audio='.wav',
               sampling_rate=None)
```
This will download and unzip both `maestro-v2.0.0.zip` and `maestro-v2.0.0-midi.zip`. You are supposed to get a folder named as maestro-v2.0.0 with maestro-v2.0.0.csv and maestro-v2.0.0.json inside.

All arugments are same as the [MAPS](#MAPS) dataset, you may want to check the avaliable arugments [here](#MAPS).

Avaliable choices for `groups` in MAESTRO are `'train', 'validation' and 'test'`

`dataset[i]` returns a dictionary containing:

```python
{'path': './maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav',
 'audio': tensor([ 0.0179,  0.0264,  0.0230,  ..., -0.1169, -0.0786, -0.0732]),
 'velocity': tensor([[...]]),
 'onset': tensor([[...]]),
 'offset': tensor([[...]]),
 'frame': tensor([[...]]),
 'sr': 44100}
```


### Getting a batch of audio segment
```python
loader = DataLoader(dataset, batch_size=4)
for batch in loader:
    audios = batch['audio'].to(device)
    frames = batch['frame'].to(device)
```

### Other functionalities
1. #### resample
```python
dataset.resample(sr, output_format='flac', num_threads=-1)
dataset = MAESTRO('./', groups=['train'], ext_audio='.wav')
```
Resample audio clips to the target sample rate `sr` and the target format `output_format`. This method requires `pydub`. After resampling, you will get flace audio clips with target sample rate.

`num_threads` sets the number of threads to use when resampling audio clips. The default value `-1` is to use all available threads. If corrupted audio clips were produced when using `num_threads>0`, set `num_threads=0` to completely disable multithreading.

