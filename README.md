# AudioDatasets
This will be a collection of PyTorch audio datasets that are not avaliable in official PyTorch yet. I am building various one-click-ready audio datasets for my research, and I hope it will also benefits other people as well. 

**Currently supported datasets:**
1. [Multilingual LibriSpeech (MLS) ](##multilingual-librispeech)

**TODO:**
1. MAPS
1. MASETRO
1. MusicNet

## Installation
pip install git+https://github.com/KinWaiCheuk/AudioDatasets.git

## Multilingual LibriSpeech
### Introduction
[Multilingual LibriSpeech (MLS)](http://www.openslr.org/94/) contains 8 langauges. This ready-to-use PyTorch dataset allows users to setup this dataset by just calling the `Multilingual LibriSpeech` class. The original dataset put all utterance labels into a single `.txt` file. For larger languages such as English, it causes a slow label loading. This custom dataset automatically sp

### Usage
To use this dataset for the first time, set `download=True`. 

```python
dataset = MultilingualLibriSpeech('../Speech', 'mls_polish_opus', 'test', download=True)
```

This will download, unzip, and split the labels. `__getitem__` returns a dictionary containing:

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

`IPA`: Default `False`. Set to `True` to extract IPA labels. Useful for phoneme recognition. Requires [phomenizer](https://github.com/bootphon/phonemizer) and [espeak](https://github.com/espeak-ng/espeak-ng). (Wokr in progress)