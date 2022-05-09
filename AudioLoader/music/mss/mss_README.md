# Music Source Separation

## FastMUSDB
### Introduction
A faster version of [MUSDB](https://github.com/sigsep/sigsep-mus-db). The dataset can be downloaded [here](https://zenodo.org/record/3338373#.Ymjj5C0RpQI).
### Usage
```python
from AudioLoader.music.mss import FastMUSDB
dataset = MusicNet(
                 root=None,
                 subsets=['train', 'test'],
                 split=None,
                 seq_duration=6.0,
                 samples_per_track=64,
                 random=False,
                 random_track_mix=False,
                 transform: Optional[Callable] = None
                 )
```

`dataset[i]` returns a tuple containing:

```python
(
    mix tensor (2,44100*seq_duration),
    source tensors (4,2,44100*seq_duration)
)
```


## MusdbHQ
### Introduction
This is a custom PyTorch Dataset Loader for MusdbHQ. This dataset can be downloaded [here](https://zenodo.org/record/3338373#.Ymjj5C0RpQI). MusdbHQ has total of 150 full-track songs. 
After you download and unzip MusdbHQ, you will get a folder calls `train` which composed of 100 songs, another folder calls `test` which composed of 50 songs.

Each song is saved as uncompressed wav files. Within each track folder, you can find the following sources: 
* mixture.wav
* drums.wav
* bass.wav
* other.wav
* vocals.wav

### Usage
To use this dataset for the first time, set `download=True`.
 
```python
from AudioLoader.music.mss import MusdbHQ
dataset = MusdbHQ(root, 
                  subset= 'training', 
                  download = True, 
                  segment=None, 
                  shift=None, 
                  normalize=True,
                  samplerate=44100, 
                  channels=2, 
                  ext=".wav")
```

This will download and unzip MusdbHQ dataset.
For the `subset` argument, you can choose from `'training_all'`, `'training'` , `'validation'`, `'test'`

* When the `subset` argument is `'training_all'`, you will get 100 songs from the train folder.
* You can set the `subset` argument to `'training'` and get 86 songs from the train folder. Set the `subset` argument to `'validation'`, you will get 14 songs from the train folder for validation purpose.

`segment` argument controls the segment length in seconds. If `None`, returns entire tracks.

`dataset[i]` returns a tensor containing:
 
```python
(4, 2, 44100*segment)
```