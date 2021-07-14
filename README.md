# AudioLoader
This will be a collection of PyTorch audio datasets that are not available in the official PyTorch dataset and torchaudio dataset yet. I am building various one-click-ready audio datasets for my research, and I hope it will also benefit other people. 

**Currently supported datasets:**
1. [Multilingual LibriSpeech (MLS) ](#multilingual-librispeech)

**TODO:**
1. [MAPS](#maps)
1. MASETRO
1. MusicNet

## Installation
`pip install git+https://github.com/KinWaiCheuk/AudioDatasets.git`

## Multilingual LibriSpeech
### Introduction
This is a custom PyTorch Dataset for Multilingual LibriSpeech (MLS).

[Multilingual LibriSpeech (MLS)](http://www.openslr.org/94/) contains 8 languages. This ready-to-use PyTorch `Dataset` class allows users to set up this dataset by just calling the `MultilingualLibriSpeech` class. The original dataset put all utterance labels into a single `.txt` file. For larger languages such as English, it causes a slow label loading. This custom `Dataset` class automatically splits the labels into smaller sizes.

### Usage
To use this dataset for the first time, set `download=True`. 

```python
dataset = MultilingualLibriSpeech('../Speech', 'mls_polish', 'train', download=True)
```

This will download, unzip, and split the labels. To download `opus` version of the dataset, simply add the suffix `_opus`. e.g. `mls_polish_opus`.

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

## MAPS
### Introduction
[MAPS](https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR?path=%2F) dataset contains 9 folders, each folder contains 30 full music recordings and the aligned midi annoations. The two folders `ENSTDkAm` and `ENSTDkCl` contains real acoustic recording obtained from a YAMAHA Disklavier. The rest are synthesized audio clips. This ready-to-use PyTorch `Dataset` class will automatically set up most of the things.

### Usage
To use this dataset for the first time, set `download=True`. 

```python
dataset = MAPS('./Folder', groups='all', download=True)
```

This will download, unzip, and extract the `.tsv` labels.

`dataset[i]` returns a dictionary containing:

```python
{'path': '../MusicDataset/MAPS/AkPnBcht/MUS/MAPS_MUS-hay_40_1_AkPnBcht.wav',
 'sr': 44100,
 'audio': tensor([[0., 0., 0.,  ..., 0., 0., 0.],
         [0., 0., 0.,  ..., 0., 0., 0.]]),
 'midi': array([[  2.078941,   2.414137,  67.      ,  52.      ],
        [  2.078941,   2.414137,  59.      ,  43.      ],
        [  2.078941,   2.414137,  55.      ,  43.      ],
        ...,
        [394.169767, 394.867987,  59.      ,  56.      ],
        [394.189763, 394.867987,  62.      ,  56.      ],
        [394.209759, 394.867987,  67.      ,  62.      ]])}
```

Each row of `midi` represents a midi note, and it contains the information: `[start_time, end_time, Midi_pitch, velocity]`.

The original audio clips are all steoro, users might want to convert them back to mono tracks first. Alternatively, the `.resample()` method can be also used to resample and convert tracks back to mono.

### Getting a batch of audio segment
To generate a batch of audio segments and piano rolls, `collect_batch(x, hop_size, sequence_length)` should be used as the `collate_fn` of PyTorch DataLoader. The `hop_size` for `collect_batch` should be same as the spectrogram hop_size, so that the piano roll obtained aligns with the spectrogram.

```python
loader = DataLoader(dataset, batch_size=4, collate_fn=lambda x: collect_batch(x, hop_size, sequence_length))
for batch in loader:
    audios = batch['audio'].to(device)
    frames = batch['frame'].to(device)
```

### Other functionalities

1. #### resample
```python
dataset.resample(sr, output_format='flac')
dataset = MAPS('./Folder', groups='all', ext_audio='.flac')
```
Resample audio clips to the target sample rate `sr` and the target format `output_format`. This method requires `pydub`. After resampling, you need to create another instance of `MAPS` in order to load the new audio files instead of the original `.wav` files.


2. #### extract_tsv
```python
dataset.extract_tsv()
```
Convert midi files into tsv files for easy loading.



## TODO
1. MusicNet
1. MAESTRO


