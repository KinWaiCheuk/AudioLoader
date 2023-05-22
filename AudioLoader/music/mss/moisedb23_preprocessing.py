from AudioLoader.music.mss import MusdbHQ
from AudioLoader.music.mss import Moisesdb23
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchaudio
from IPython.display import Audio
import tqdm
import math
from pathlib import Path
%load_ext autoreload
%autoreload 2

# downsampling & meta data

dataset_path = Path('/root/dataset/moisesdb23_labelnoise_v1.0')
new_dataset_path = '/root/dataset/moisesdb23_labelnoise_v1.0_16k'

name_sr_dict = {}
for i in tqdm.tqdm(list(dataset_path.iterdir())):
    for source_name in ['bass', 'drums', 'other', 'vocals']:
        source_path = os.path.join(i,  source_name + '.wav')
        wav, sr = torchaudio.load(source_path)
        # wav = wav.mean(0, keepdim=True) # combine two channels into one channels
        wav = torchaudio.functional.resample(wav, sr, 16000)
        Path(os.path.join(new_dataset_path, i.stem)).mkdir(parents=True, exist_ok=True) # create the folder if not exists
        new_path = os.path.join(new_dataset_path, i.stem, source_name + '.flac')
        
        torchaudio.save(new_path, wav, 16000)
        name_sr_dict[i.stem] = wav.shape[1]
        
        
with open('moisesdb23_meta_16k.pkl', 'wb') as f:
    pickle.dump(name_sr_dict, f)