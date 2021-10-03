import os, torch, torchaudio 
import torch.utils.data as data
import torchvision.transforms as transforms

from sklearn.utils import shuffle
import numpy as np
import pandas as pd

from variables import *

def calculate_audio_length(sampling_rate, audio_signal):
    audio_length = audio_signal.shape[0] / sampling_rate
    return audio_length

def object_to_tensor(arr, device):
    arr_new = None
    for elem in arr:
        elem = torch.from_numpy(elem)
        elem = torch.unsqueeze(elem, 0)
        if arr_new is None:
            arr_new = elem
        else:
            arr_new = torch.cat((arr_new, elem), dim=0)
    return arr_new.to(device)

#custom audio dataset
class SpeechEnhancementData(data.Dataset):
    def __init__(self):
        self.audio_dir = audio_dir
        self.data, self.class_dict = self.create_audio_dataframe()
        self.data = shuffle(self.data)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.resample_rate = resample_rate
        self.signal_length = signal_length

        # Extraction & Transformations
        self.is_resample = True # resample audio signals since all samples doesn't have same sample rate
        self.is_mix_down = True # mix down to mono (Signals with more than one channel flatten to one-dimensional)           

        self.pre_padding = False  
        self.pre_truncation = False 

        self.is_mel_transform = True   

    def ETLaudio(self, row):
        file_path = row['audio_file']
        audio_signal, sampling_rate = torchaudio.load(file_path)
        audio_signal = audio_signal.to(self.device)
        audio_length = calculate_audio_length(sampling_rate, audio_signal)
        row['audio_length'] = audio_length

        ########################## Resample ########################################
        if self.is_resample:
            if sampling_rate != self.resample_rate:
                resample_audio_signal = torchaudio.transforms.Resample(
                                                                sampling_rate, 
                                                                self.resample_rate
                                                                ).to(self.device)
                audio_signal = resample_audio_signal(audio_signal)

        ########################## Multi Dimension to MONO #########################
        if self.is_mix_down:
            if audio_signal.shape[0] > 1:
                audio_signal = torch.mean(audio_signal, dim=0, keepdim=True)

        ########################## Padding / Truncation ############################ 
            channels, frames = audio_signal.shape
            if frames < self.signal_length:
                if self.pre_padding:
                    audio_signal = torch.cat((torch.zeros(channels, self.signal_length - frames), audio_signal), dim=1)
                else:
                    audio_signal = torch.cat(audio_signal, (torch.zeros(channels, self.signal_length - frames)), dim=1)
    
            else:
                if self.pre_truncation:
                    audio_signal = audio_signal[:, -self.signal_length :]
                else:
                    audio_signal = audio_signal[:, :self.signal_length]

        ########################## MEL Spectrogram######## #########################
        if self.is_mel_transform:
            mel_transform = transforms.Compose([
                                        torchaudio.transforms.MelSpectrogram(
                                                                    sample_rate=resample_rate, 
                                                                    n_fft=frame_size, 
                                                                    hop_length=hop_length, 
                                                                    n_mels=n_mels).to(self.device)
                                                    ])   
            audio_signal = mel_transform(audio_signal)    

        row['audio_signal'] = audio_signal.cpu().detach().numpy()
        return row

    def load_audio_files(self):
        audio_classes = os.listdir(self.audio_dir)
        audio_files = []
        aduio_labels = []
        class_dict = {}
        for i, audio_class in enumerate(audio_classes):
            class_dict[audio_class] = i
            audio_class_dir = os.path.join(self.audio_dir, audio_class)
            audio_files_in_class = os.listdir(audio_class_dir)
            for audio_file in audio_files_in_class:
                audio_file_path = os.path.join(audio_class_dir, audio_file)
                if audio_file_path.endswith('.wav'):
                    audio_files.append(audio_file_path)
                    aduio_labels.append(audio_class)
                
        return audio_files, aduio_labels, class_dict

    def create_audio_dataframe(self):
        audio_files, audio_labels, class_dict = self.load_audio_files()
        audio_df = pd.DataFrame(audio_files, columns=['audio_file'])
        audio_df['audio_label'] = audio_labels
        return audio_df, class_dict

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_df = self.data.iloc[index * batch_size : (index + 1) * batch_size]
        audio_df = audio_df.apply(self.ETLaudio, axis=1)
        audio_df = audio_df.reset_index(drop=True)
        audio_signal = audio_df['audio_signal'].values
        audio_label = audio_df['audio_label'].values
        audio_signal = object_to_tensor(audio_signal, self.device)
        audio_label = torch.as_tensor([self.class_dict[label] for label in audio_label])
        return audio_signal, audio_label


# a = SpeechEnhancementData()
# audio_signal, audio_label = a.__getitem__(0)
# print(audio_signal.shape)