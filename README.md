# AudioLoader
This will be a collection of PyTorch audio dataset loaders that are not available in the official PyTorch dataset and torchaudio dataset yet. I am building various one-click-ready audio datasets for my research, and I hope it will also benefit other people. 

**Currently supported datasets:**
1. [Speech](./AudioLoader/speech/speech_README.md#Speech)
    1. [Multilingual LibriSpeech (MLS)](./AudioLoader/speech/speech_README.md#Multilingual-LibriSpeech)
    1. [TIMIT](./AudioLoader/speech/speech_README.md#TIMIT)
    1. [SpeechCommands v2 (12 classes)](./AudioLoader/speech/speech_README.md#SpeechCommandsv2)
1. [Automatic Music Transcription (AMT)](./AudioLoader/music/amt_README.md#Automatic-Music-Transcription)
    1. [MAPS](./AudioLoader/music/amt_README.md#MAPS)
    1. [MusicNet](./AudioLoader/music/amt_README.md#MusicNet)
1. [Music Source Separation (MSS)](./AudioLoader/music/mss/mss_README.md#Music-Source-Separation)
    1. [FastMUSDB](./AudioLoader/music/mss/mss_README.md#FastMUSDB)
    1. [MusdbHQ](./AudioLoader/music/mss/mss_README.md#MusdbHQ)

**TODO:**
1. MASETRO

## Installation
`pip install git+https://github.com/KinWaiCheuk/AudioLoader.git`

## News & Changelog
**version 0.0.3** (10 Sep 2021): 
1. Replace broken links with a working links for `MAPS` and `TIMIT`
1. Remove the slience indicators in the phonemic labels for TIMIT

