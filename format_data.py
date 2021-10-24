import os
import librosa
import numpy as np
from tqdm import tqdm

from variables import *
 
def slice_signal(file, window_size=window_size, stride=stride, sample_rate=sample_rate):
    """
    This utility function slices the audio signal into overlapping windows.
    The Reason for this windowing technique is to reduce the end-point discontinuouty of the signal.

    Args:
        file: the path to the audio file
        window_size: the size of the window
        stride: the stride of the window
        sample_rate: the sample rate of the audio file

    Returns:
        sliced_signal: a list of sliced signals

    Note:
        default stride is 0.5, which means the hop length is half of the window size.
        stride is 1 means no overlap.
    """
    try:
        audio_signal = librosa.load(file, sr=sample_rate)[0]
    except:
        print('Error loading file {}'.format(file))
        assert False

    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(audio_signal), hop):
        start_idx = end_idx - window_size
        start_idx = max(0, start_idx)

        slice_sig = audio_signal[start_idx:end_idx]
        slices.append(slice_sig)
    slices = np.array(slices)
    return slices

def serialize_data(clean_dir, noisy_dir, serialized_dir, stride, split):
    """
    Serialize the sliced signals and save on separate dir.

    Args:
        clean_dir: the path to the clean audio files
        noisy_dir: the path to the noisy audio files
        serialized_dir: the path to the serialized data
        stride: the stride of the window
        split: the split of the data

    Returns:
        None
    """
    files = os.listdir(clean_dir)
    assert len(files) > 0, 'No files found in {}'.format(clean_dir)
    for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(split)):
        clean_file = os.path.join(clean_dir, filename)
        noisy_file = os.path.join(noisy_dir, filename)

        # slice both clean signal and noisy signal
        clean_sliced = slice_signal(clean_file, window_size, stride, sample_rate)
        noisy_sliced = slice_signal(noisy_file, window_size, stride, sample_rate)

        # serialize - file format goes [original_file]_[slice_number].npy
        for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
            pair = np.array([slice_tuple[0], slice_tuple[1]])
            np.save(os.path.join(serialized_dir, '{}_{}'.format(filename, idx)), arr=pair)

def process_and_serialize(split, stride=stride):
    """
    Apply pre-processing to the data and serialize it.
    """
    clean_dir = clean_train_dir if split == 'train' else clean_test_dir
    noisy_dir = noisy_train_dir if split == 'train' else noisy_test_dir
    serialized_dir = serialized_train_dir if split == 'train' else serialized_test_dir

    serialize_data(clean_dir, noisy_dir, serialized_dir, stride, split)

def data_verify(split):
    """
    Verifies the length of each data after pre-process.
    """
    serialized_dir = serialized_train_dir if split == 'train' else serialized_test_dir
    files = os.listdir(serialized_dir)
    
    for filename in tqdm(files, desc='Verify serialized {} audios'.format(split)):
        data_pair = np.load(os.path.join(serialized_dir, filename))
        if data_pair.shape[1] != window_size:
            print('Snippet length not {} : {} instead in {}'.format(window_size, data_pair.shape[1], data_pair))
            break

def preprocess_data():
    """
    Preprocess train and test data.
    """
    process_and_serialize("train")
    data_verify("train")

    process_and_serialize("test")
    data_verify("test")

if __name__ == '__main__':
    preprocess_data()