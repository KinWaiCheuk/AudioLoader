# Speech

## Multilingual LibriSpeech
### Introduction
This is a custom PyTorch Dataset Loader for Multilingual LibriSpeech (MLS).

[Multilingual LibriSpeech (MLS)](http://www.openslr.org/94/) contains 8 languages. This ready-to-use PyTorch `Dataset` allows users to set up this dataset by just calling the `MultilingualLibriSpeech` class. The original dataset put all utterance labels into a single `.txt` file. For larger languages such as English, it causes a slow label loading. This custom `Dataset` class automatically splits the labels into smaller sizes.

### Usage
To use this dataset for the first time, set `download=True`. 

```python
from AudioLoader.speech import MultilingualLibriSpeech
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

## TIMIT
### Introduction
This is a custom PyTorch Dataset Loader for [The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/LDC93S1). This dataset can be downloaded via this [github repository](https://github.com/philipperemy/timit).
### Usage
To use this dataset for the first time, set `download=True`. 

```python
from AudioLoader.speech import TIMIT
dataset = TIMIT('./YourFolder',
                split='train',
                groups='all',
                download=True)
```

This will download, unzip, and split the labels inside `YourFolder`. You can control which dialect regions to load via the `groups` argument. `gourps='all'` loads all dialect regions; `groups=[1,5]` loads `DR1` and `DR5`.

`dataset[i]` returns a dictionary containing:

```python
{'path': 'TIMIT/data/TRAIN/DR1/FSMA0/SX361.WAV.wav',
 'waveform': tensor([[3.0518e-05, 1.5259e-04, 6.1035e-05,  ..., 1.5259e-04, 0.0000e+00,
          2.1362e-04]]),
 'sample_rate': 16000,
 'DR': 'DR1',
 'gender': 'F',
 'speaker_id': 'SMA0',
 'phonemics': 'dh ix s pcl p iy tcl ch s ix m pcl p ow z y ix m ay q bcl b iy gcl g ih n m ah n dcl d ey ',
 'words': 'the speech symposium might begin monday '}
```

## SpeechCommandsv2
### Introduction
This is a custom PyTorch Dataset Loader for SpeechCommands version 2 with 12 classes. 

Original SpeechCommands version 2 has total 35 single wordings. 10 out of 35 words are chosen.
* Class 1 to 10 (following the class order): `‘down’, ‘go’, ‘left’, ‘no’, ‘off’, ‘on’, ‘right’, ‘stop’, ‘up’, ‘yes’` 
* Class 11: `class ‘unknown’`  represents the remaining 25 unchosen words.
* Class 12: `class ‘silence’` represent no word can be detected which is created from background noise.

### Usage
To use this dataset for the first time, set `download=True`.

```python
from AudioLoader.speech import SPEECHCOMMANDS_12C
dataset = SPEECHCOMMANDS_12C('./YourFolder','speech_commands_v0.02',
'SpeechCommands',download=True ,subset= 'training')
```

This will download, unzip, and split the labels inside YourFolder. To download validation set or test set, simply change `subset` argument to `validation` or `testing` respectively

`dataset[i]` returns a tuple containing:
```python
(waveform, (sample_rate, label, speaker_id, utterance_number))
```
