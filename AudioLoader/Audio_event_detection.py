
# import requests
# from tqdm import tqdm
# import tarfile

from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from typing import Tuple, Union
from torch import Tensor
import torch
import os
import time
import json
import tqdm
import shutil
import glob
import multiprocessing as mp
import warnings
from distutils.dir_util import copy_tree
from torchaudio.compliance import kaldi # for downsampling
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
import hashlib
import json
import subprocess


class DCASE2016(Dataset):
    """Dataset class for DCASE2016 dataset.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
            
        split (str):
            Choose different dataset splits such as ``"train"``, or ``"test"`` or ``"dev"``. (default: ``"train"``).
        
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        
    """
    

    def __init__(self,
                 root: Union[str, Path],
                 split: str = "train",                 
                 download: bool = False
                ):
        self.split = split
        if self.split == "train" or self.split =="dev":
            url = "https://archive.org/compress/dcase2016_task2_train_dev/formats=ZIP&file=/dcase2016_task2_train_dev.zip"
        else:
            url = "https://archive.org/download/dcase2016_task2_test_public/dcase2016_task2_test_public.zip"
        folder_name = 'DCASE2016'
        
        download_path = os.path.join(root, folder_name)
        assert self.split.upper()=="TRAIN" or self.split.upper()=="TEST" or self.split.upper()=="DEV", f"split={split} is not present in this dataset"
        
        if self.split =="train":
            archive_name = f'dcase2016_task2_train_dev.zip'
            self._path = os.path.join(download_path, f"dcase2016_task2_train_dev",f"dcase2016_task2_train") 
        elif self.split =="dev":
            archive_name = f'dcase2016_task2_train_dev.zip'
            self._path = os.path.join(download_path, f"dcase2016_task2_train_dev",f"dcase2016_task2_dev","sound") 
        elif self.split == "test":
            archive_name = f'dcase2016_task2_test_public.zip'
            self._path = os.path.join(download_path, f"dcase2016_task2_test_public", "sound") 
        
        checksum_dict = {"test": "ac98768b39a08fc0c6c2ddd15a981dd7", "train":"0eab0635cff8e2d76e9e38b7a7de342d", "dev":"0eab0635cff8e2d76e9e38b7a7de342d"} #same value for train&dev

        if download: #bad data format==> unzipped file has same name as zipped file, create tmp path and remove as a workaround
            #file exists and extracted
            if os.path.isfile(os.path.join(download_path,archive_name)) and os.path.exists(self._path):
                print(f"Dataset archive exists, all files are extracted. Using all file from {self._path} ")
            #file exists but not extracted
            if os.path.isfile(os.path.join(download_path,archive_name)) and not os.path.exists(self._path):
                print(f"Dataset archive exists, extracting archive:{os.path.join(download_path,archive_name)}")
                if self.split!="test":
                    tmp_path = os.path.join(download_path,"tmp")
                    extract_archive(os.path.join(download_path, archive_name),to_path = tmp_path)
                    extract_archive(os.path.join(tmp_path, archive_name),to_path = download_path)
                    os.system(f"rm -rf {tmp_path}")
                else:
                    extract_archive(os.path.join(download_path, archive_name))
                print(f"Using all file from {self._path} ")
            #file not exist
            elif not os.path.isfile(os.path.join(download_path,archive_name)):
                print(f"archive {os.path.join(download_path,archive_name)} not exists, try downloading")
                if not os.path.exists(download_path):
                    os.makedirs(download_path)
                try:
                    download_url(url, download_path) #, hash_value = checksum_dict[self.split], hash_type = "md5"
                    if self.split!="test":
                        tmp_path = os.path.join(download_path,"tmp")
                        extract_archive(os.path.join(download_path, archive_name),to_path = tmp_path)
                        extract_archive(os.path.join(tmp_path, archive_name),to_path = download_path)
                        os.system(f"rm -rf {tmp_path}")
                    else:
                        extract_archive(os.path.join(download_path, archive_name))
                    print(f"All files are extracted. Using all file from {self._path} ")

                except:
                    raise Exception('Auto download fails. '+
                                    'You may want to download it manually from:\n'+
                                    url+ '\n' +
                                    f'Then, put it inside {download_path}')   
        else:   
            #archive is downloaded and extracted
            if os.path.isfile(os.path.join(download_path, archive_name)) and not os.path.exists(self._path):
                print(f"Dataset archive exists, all files are extracted. Using all file from {self._path} ")
            #archive is downloaded but not extracted
            elif os.path.isfile(os.path.join(download_path, archive_name)) and not os.path.exists(self._path):
                print(f'archive:{os.path.join(download_path, archive_name)} exists, extracting...')
                if self.split!="test":
                    tmp_path = os.path.join(download_path,"tmp")
                    extract_archive(os.path.join(download_path, archive_name),to_path = tmp_path)
                    extract_archive(os.path.join(tmp_path, archive_name),to_path = download_path)
                    os.system(f"rm -rf {tmp_path}")
                else:
                    extract_archive(os.path.join(download_path, archive_name))
                print(f"Using all file from {self._path} ")
            else:
                raise FileNotFoundError(f"Dataset not found at {self._path}, please specify the correct location or set `download=True`")
                        
        self._walker = glob.glob(f"{self._path}/*.wav") 
        self.label_dict = {'clearthroat':0, 'cough':1, 'doorslam':2, 'drawer':3, 'keyboard':4, 'keys':5, 'knock':6, 'laughter':7, 'pageturn':8, 'phone':9, 'speech':10} #keys == keysDrop; convert keysDrop in trainset to keys


    def __getitem__(self, n):
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tuple: ``(path, waveform, sample_rate, start_end_event)``
        """
        file_path = self._walker[n]
        waveform, sample_rate = torchaudio.load(file_path)

        if self.split!="train":  
            label_path = os.path.dirname(os.path.dirname(file_path)) +"/annotation/"+ os.path.basename(file_path).split(".")[0]+".txt"
            lst_of_events = [e[:-1].split("\t") for e in open(label_path,"r").readlines()] #read lines, remove "\n", split
            lst_of_events_encoded = [(float(x[0]), float(x[1]), self.label_dict[x[2]]) for x in lst_of_events] #lists of (start, end, class)
        else: 
            label_name = os.path.basename(file_path).split(".")[0][:-3]
            if label_name == "keysDrop":
                label_name = "keys" #check readme.txt, and http://dcase.community/challenge2016/task-sound-event-detection-in-synthetic-audio; keysDrop and keys are equivalent
            lst_of_events_encoded= [("dummy_start","dummy_end",label_name)]
        batch = {'path':file_path,
                 'waveform': waveform,
                 'sample_rate': sample_rate,
                 'start_end_event':lst_of_events_encoded
                }

        return batch
    
    def __len__(self) -> int:
        return len(self._walker)