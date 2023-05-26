import torch
import numpy as np
import time
import urllib
import urllib.request
import tarfile
import threading
import zipfile
import os
from typing import Any, Iterable, List, Optional

def tsv2roll(tsv, audio_length, sample_rate, hop_size, max_midi, min_midi):
    """
    Converting a tsv file into a piano roll
    
    Parameters
    ----------
    tsv : numpy.ndarray
        The tsv label in the array format
    hop_size: int
        Hop size for the spectrogram. It will be used to convert tsvs into piano rolls
    max_midi: int
        The top bin of the pianoroll. Default 108 which corresponds to C8, the highest key on a standard 88-key piano
    min_midi: int
        The lowest bin of the pianoroll. Default 21 which corresponds to A0, the lowest key on a standard 88-key piano
    """
    
    
    n_keys = max_midi - min_midi + 1 # Calutate number of bins for the piano roll
    n_steps = (audio_length - 1) // hop_size + 1 # Calulate number of timesteps for the piano roll
    
    pianoroll = torch.zeros((n_steps, n_keys), dtype=int)
    velocity_roll = torch.zeros((n_steps, n_keys), dtype=int)
    
    for onset, offset, note, vel in tsv:
        left = int(round(onset * sample_rate / hop_size)) # Convert time to time step
        onset_right = min(n_steps, left + 1) # Ensure the time step of onset would not exceed the last time step
        frame_right = int(round(offset * sample_rate / hop_size))
        frame_right = min(n_steps, frame_right) # Ensure the time step of frame would not exceed the last time step
        offset_right = min(n_steps, frame_right + 1)

        f = int(note) - min_midi
        pianoroll[left:onset_right, f] = 3 # assigning onset
        pianoroll[onset_right:frame_right, f] = 2 # assigning sustain
        pianoroll[frame_right:offset_right, f] = 1 # assigning offset
        velocity_roll[left:frame_right, f] = vel    
        
        
        
    return pianoroll, velocity_roll




def check_md5(path, md5_hash):
    with open(path, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()

        assert md5_returned==md5_hash, f"{os.path.basename(path)} is corrupted, please download it again"
        
# This part of the code is obtained from torchaudio==0.9
# https://github.com/pytorch/audio/blob/a85b2398722182dd87e76d9ffcbbbf7e227b83ce/torchaudio/datasets/utils.py
        
def stream_url(url: str,
               start_byte: Optional[int] = None,
               block_size: int = 32 * 1024,
               progress_bar: bool = True) -> Iterable:
    """Stream url by chunk

    Args:
        url (str): Url.
        start_byte (int, optional): Start streaming at that point (Default: ``None``).
        block_size (int, optional): Size of chunks to stream (Default: ``32 * 1024``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
    """

    # If we already have the whole file, there is no need to download it again
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as response:
        url_size = int(response.info().get("Content-Length", -1))
    if url_size == start_byte:
        return

    req = urllib.request.Request(url)
    if start_byte:
        req.headers["Range"] = "bytes={}-".format(start_byte)

    with urllib.request.urlopen(req) as upointer, tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=url_size,
            disable=not progress_bar,
    ) as pbar:

        num_bytes = 0
        while True:
            chunk = upointer.read(block_size)
            if not chunk:
                break
            yield chunk
            num_bytes += len(chunk)
            pbar.update(len(chunk))
        
def download_url(url: str,
                 download_folder: str,
                 filename: Optional[str] = None,
                 hash_value: Optional[str] = None,
                 hash_type: str = "sha256",
                 progress_bar: bool = True,
                 resume: bool = False) -> None:
    """Download file to disk.

    Args:
        url (str): Url.
        download_folder (str): Folder to download file.
        filename (str, optional): Name of downloaded file. If None, it is inferred from the url (Default: ``None``).
        hash_value (str, optional): Hash for url (Default: ``None``).
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
        resume (bool, optional): Enable resuming download (Default: ``False``).
    """

    req = urllib.request.Request(url, method="HEAD")
    req_info = urllib.request.urlopen(req).info()

    # Detect filename
    filename = filename or req_info.get_filename() or os.path.basename(url)
    filepath = os.path.join(download_folder, filename)
    if resume and os.path.exists(filepath):
        mode = "ab"
        local_size: Optional[int] = os.path.getsize(filepath)

    elif not resume and os.path.exists(filepath):
        raise RuntimeError(
            "{} already exists. Delete the file manually and retry.".format(filepath)
        )
    else:
        mode = "wb"
        local_size = None

    if hash_value and local_size == int(req_info.get("Content-Length", -1)):
        with open(filepath, "rb") as file_obj:
            if validate_file(file_obj, hash_value, hash_type):
                return
        raise RuntimeError(
            "The hash of {} does not match. Delete the file manually and retry.".format(
                filepath
            )
        )

    with open(filepath, mode) as fpointer:
        for chunk in stream_url(url, start_byte=local_size, progress_bar=progress_bar):
            fpointer.write(chunk)

    with open(filepath, "rb") as file_obj:
        if hash_value and not validate_file(file_obj, hash_value, hash_type):
            raise RuntimeError(
                "The hash of {} does not match. Delete the file manually and retry.".format(
                    filepath
                )
            )
            
def extract_archive(from_path: str, to_path: Optional[str] = None, overwrite: bool = False) -> List[str]:
    """Extract archive.
    Args:
        from_path (str): the path of the archive.
        to_path (str, optional): the root path of the extraced files (directory of from_path) (Default: ``None``)
        overwrite (bool, optional): overwrite existing files (Default: ``False``)

    Returns:
        list: List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchaudio.datasets.utils.download_from_url(url, from_path)
        >>> torchaudio.datasets.utils.extract_archive(from_path, to_path)
    """

    if to_path is None:
        to_path = os.path.dirname(from_path)

    try:
        with tarfile.open(from_path, "r") as tar:
            logging.info("Opened tar file {}.".format(from_path))
            files = []
            for file_ in tar:  # type: Any
                file_path = os.path.join(to_path, file_.name)
                if file_.isfile():
                    files.append(file_path)
                    if os.path.exists(file_path):
                        logging.info("{} already extracted.".format(file_path))
                        if not overwrite:
                            continue
                tar.extract(file_, to_path)
            return files
    except tarfile.ReadError:
        pass

    try:
        with zipfile.ZipFile(from_path, "r") as zfile:
            logging.info("Opened zip file {}.".format(from_path))
            files = zfile.namelist()
            for file_ in files:
                file_path = os.path.join(to_path, file_)
                if os.path.exists(file_path):
                    logging.info("{} already extracted.".format(file_path))
                    if not overwrite:
                        continue
                zfile.extract(file_, to_path)
        return files
    except zipfile.BadZipFile:
        pass

    raise NotImplementedError("We currently only support tar.gz, tgz, and zip achives.")