# AudioLoader
AudioLoader is a PyTorch dataset based on [torchaudio](https://pytorch.org/audio/stable/datasets.html). It contains a collection of datasets that are not available in [torchaudio](https://pytorch.org/audio/stable/datasets.html) yet.

**Currently supported datasets:**
1. [Speech](./AudioLoader/speech/speech_README.md#Speech)
    1. [Multilingual LibriSpeech (MLS)](./AudioLoader/speech/speech_README.md#Multilingual-LibriSpeech)
    1. [TIMIT](./AudioLoader/speech/speech_README.md#TIMIT)
    1. [SpeechCommands v2 (12 classes)](./AudioLoader/speech/speech_README.md#SpeechCommandsv2)
1. [Automatic Music Transcription (AMT)](./AudioLoader/music/amt_README.md#Automatic-Music-Transcription)
    1. [MAPS](./AudioLoader/music/amt_README.md#maps)
    1. [MusicNet](./AudioLoader/music/amt_README.md#musicnet)
    1. [MAESTRO](./AudioLoader/music/amt_README.md#maestro)
1. [Music Source Separation (MSS)](./AudioLoader/music/mss/mss_README.md#Music-Source-Separation)
    1. [FastMUSDB](./AudioLoader/music/mss/mss_README.md#FastMUSDB)
    1. [MusdbHQ](./AudioLoader/music/mss/mss_README.md#MusdbHQ)
    
## Example code
A complete example code is available in this [repository](https://github.com/KinWaiCheuk/pytorch_template). The following pseudo  code shows the general idea of how to apply AudioLoader to your existing code.

```python
from AudioLoader.speech import TIMIT
from torch.utils.data import DataLoader

# AudioLoader helps you to set up supported datasets
dataset = TIMIT('./YourFolder',
                split='train',
                groups='all',
                download=True)
train_loader = DataLoader(dataset,
                          batch_size=4)

# Pass the dataset to you 
model = MyModel()
trainer = pl.Trainer()
trainer.fit(model, train_loader)

```

## Installation
`pip install git+https://github.com/KinWaiCheuk/AudioLoader.git`

## News & Changelog
**version 0.0.3** (10 Sep 2021): 
1. Replace broken links with a working links for `MAPS` and `TIMIT`
1. Remove the slience indicators in the phonemic labels for TIMIT

