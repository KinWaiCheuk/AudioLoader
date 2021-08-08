# AudioLoader
This will be a collection of PyTorch audio dataset loaders that are not available in the official PyTorch dataset and torchaudio dataset yet. I am building various one-click-ready audio datasets for my research, and I hope it will also benefit other people. 

**Currently supported datasets:**
1. [Multilingual LibriSpeech (MLS) ](#multilingual-librispeech)
1. [TIMIT](#The-DARPA-TIMIT-Acoustic-Phonetic-Continuous-Speech-Corpus)
1. [MAPS](#maps)
1. [MusicNet](#MusicNet)

**TODO:**
1. MASETRO

## Installation
`pip install git+https://github.com/KinWaiCheuk/AudioLoader.git`

## Multilingual LibriSpeech
### Introduction
This is a custom PyTorch Dataset Loader for Multilingual LibriSpeech (MLS).

[Multilingual LibriSpeech (MLS)](http://www.openslr.org/94/) contains 8 languages. This ready-to-use PyTorch `Dataset` allows users to set up this dataset by just calling the `MultilingualLibriSpeech` class. The original dataset put all utterance labels into a single `.txt` file. For larger languages such as English, it causes a slow label loading. This custom `Dataset` class automatically splits the labels into smaller sizes.

### Usage
To use this dataset for the first time, set `download=True`. 

```python
from AudioLoader.Speech import MultilingualLibriSpeech
dataset = MultilingualLibriSpeech('./YourFolder', 'mls_polish', 'train', download=True)
```

This will download, unzip, and split the labels inside `YourFolder`. To download `opus` version of the dataset, simply add the suffix `_opus`. e.g. `mls_polish_opus`.

`dataset[i]` returns a dictionary containing:

```python
{'path': '../Speech/mls_polish_opus/test/audio/8758/8338/8758_8338_000066.opus',
 'waveform': tensor([[ 1.8311e-04,  1.5259e-04,  1.5259e-04,  ...,  1.5259e-04,
           9.1553e-05, -3.0518e-05]]),
 'sample_rate': 48000,
 'utterance': 'i zaczynają z wielką ostrożnością rozdzierać jedwabistą powłokę w tem miejscu gdzie się znajduje głowa poczwarki gdyż młoda mrówka tak jest niedołężną że nawet wykluć się ze swego więzienia nie może bez obcej pomocy wyciągnąwszy ostrożnie więźnia który jest jeszcze omotany w rodzaj pieluszki',
 'speaker_id': 8758,
 'chapter_id': 8338,
 'utterance_id': 66}
```

### Other functionalities

1. #### extract_limited_train_set
```python
dataset.extract_limited_train_set()
```
It extracts the `9hr` and `1hr` train sets into a new folder called `limited_train`. It would be useful for researchers who work on low-resource training.


2. #### extract_labels
```python
dataset.extract_labels(split_name, num_threads=0, IPA=False)
```
It splits the single text label `.txt` file into smaller per chapter `.txt` files. It dramastically improves the label loading efficiency. When setting up the dataset for the first time, `self.extract_labels('train')`, `self.extract_labels('dev')`, and `self.extract_labels('test')` are called automaically.

`split_name`: `train`, `dev`, `test`, `limited_train`

`num_threads`: Default `0`. Determine how many threads are used to split the labels. Useful for larger dataset like English.

`IPA`: Default `False`. Set to `True` to extract IPA labels. Useful for phoneme recognition. Requires [phomenizer](https://github.com/bootphon/phonemizer) and [espeak](https://github.com/espeak-ng/espeak-ng).

## The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus
### Introduction
This is a custom PyTorch Dataset Loader for [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1). This dataset can be downloaded via this [github repository](https://github.com/philipperemy/timit).
### Usage
To use this dataset for the first time, set `download=True`. 

```python
from AudioLoader.Speech import TIMIT
dataset = TIMIT('./YourFolder',
                split='train',
                groups='all',
                download=True)
```

This will download, unzip, and split the labels inside `YourFolder`. You can control which dialect regions to load via the `groups` argument. `gourps='all'` loads all dialect regions; `groups=[1,5]` loads `DR1` and `DR5`.

`dataset[i]` returns a dictionary containing:

```python
{'path': '../../SpeechDataset/TIMIT/data/TRAIN/DR1/MTJS0/SX292.WAV.wav',
 'waveform': tensor([[ 7.0190e-04, -1.8311e-04, -3.0518e-05,  ...,  6.1035e-05,
           1.2207e-04, -3.0518e-04]]),
 'sample_rate': 16000,
 'DR': 'DR1',
 'gender': 'M',
 'speaker_id': 'TJS0',
 'phonemics': ['h#',...'z','h#'],
 'words': ['these',...,'all','times']}
```



## MAPS
### Introduction
[MAPS](https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR?path=%2F) dataset contains 9 folders, each folder contains 30 full music recordings and the aligned midi annoations. The two folders `ENSTDkAm` and `ENSTDkCl` contains real acoustic recording obtained from a YAMAHA Disklavier. The rest are synthesized audio clips. This ready-to-use PyTorch `Dataset` will automatically set up most of the things.

### Usage
To use this dataset for the first time, set `download=True`. 

```python
from AudioLoader.Music import MAPS
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
               ext_audio='.wav')
```

This will download, unzip, and extract the `.tsv` labels.

If `use_cache=True`, the output dictionary containing `(path, audio, velocity, onset, offset, frame, sr)` will be saved as a `.pt` file. Loading from `.pt` files is slightly faster.

If `preload=True`, the whole dataset will be loaded into RAM, which allows a faster data accessing for each iteration. If the dataset is too large to be loaded in RAM, you can set `preload=False`

If `sequence_length=None`, it will load the full audio track, otherwise, the output will be automatically crop to `sequence_length`. If `hop_length` is set to be same as the one used in spectrogram extraction, the piano rolls returned by this dataset will be aligned with the spectrogram. 

`groups` controls which folders in MAPS to be loaded. `groups='all'` loads all folders; `groups='train'` loads the train set; `groups=test` loads the test set. Alternatively, you can also pass a list of the folder name `['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD']` to load the folders you want.

`ext_audio`: If you have resampled your audio into `flac` format, you can use this to control which set of audio to load.

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

3. ### clear_caches
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
from AudioLoader.Music import MusicNet
musicnet_dataset = MusicNet('./',
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

All arugments are same as the [MAPS](#maps) dataset, you may want to check the avaliable arugments [here](#maps).

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

Same as [MAPS](#maps)




