import torch
import torchaudio
import torch.nn as nn
import torchaudio.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob
from tqdm.notebook import tqdm

class MelDataset(Dataset):
    def __init__(self, mel_params):
        is_cuda = True and torch.cuda.is_available()
        cpu = torch.device('cpu')
        gpu = torch.device('cuda')
        device = gpu if is_cuda else cpu
        print(f"using device {device}")
        torch.cuda.empty_cache()
        
        
        self.traint = pd.read_csv( 'data/train_tp.csv' )
        self.trainf = pd.read_csv( 'data/train_fp.csv' )
        self.files = glob.glob( 'data/train/*.flac' )
        
        self.traint['val'] = 1
        self.trainf['val'] = 0
        
        self.train = pd.concat([self.traint, self.trainf])
        
        self.audiofiles = self.traint.recording_id.unique()

        del self.traint
        del self.trainf
        
        # convert all train audio files into Mel Spec
        
        self.mel_specs = []
        self.species = []
        
        for id in tqdm(self.audiofiles):
            waveform, sr = torchaudio.load(f"data/train/{id}.flac")
            waveform = waveform.to(device)
            self.mel_specs.append(transforms.MelSpectrogram(
                sample_rate=mel_params['sample_rate'], 
                n_mels=mel_params['n_mels'], 
                n_fft=mel_params['n_fft'], 
                f_min=mel_params['f_min'], 
                f_max=mel_params['f_max'],
                power=mel_params['power']
            ).to(device)(waveform))
            
            del waveform
            torch.cuda.empty_cache()
            
            subset = self.train[self.train['recording_id'] == id]
            cur_label = torch.zeros(24).to(device)
            
            #For each mel spec, write which species are in it.
            for index, entry in subset.iterrows():
                cur_label[entry.species_id] = entry.val
            
            self.species.append(cur_label)
        
        torch.cuda.empty_cache()
        
    def get_recording_ids(self):
        return self.audiofiles
    def __len__(self):
        return len(self.mel_specs)
    def __getitem__(self, idx):
        return (self.mel_specs[idx], self.species[idx], self.audiofiles[idx])


class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()

        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,       # e.g. (batch, time_step, input_size)
        )        
        
#         self.rnn2 = nn.LSTM(
#             hidden_size=64,
#             num_layers=1,
#             batch_first=True,
#         )

        self.l1 = nn.Linear(64, 64)
        self.d1 = nn.Dropout(p=0.2)
        self.out = nn.Linear(64, 24)
        self.relu = nn.ReLU()
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn1(x, None)
#         x = self.rnn2(x)
        x = F.relu(self.l1(r_out[:, -1, :]))
#         x = self.relu(self.l1(x))
        x = self.d1(x)
        x = F.sigmoid(self.out(x))
        return x

class Params():
    SAMPLE_RATE= 48000
    BATCH_SIZE = 16
    VALIDATION_SPLIT = .2
    SHUFFLE = True
    RANDOM_SEED = 42
    LR = 0.001
    mel_params = {
        "sample_rate": SAMPLE_RATE,
        "n_fft": 4096,
        "hop_length": 2048,
        "n_mels": 64,
        "f_min": 0,
        "f_max": SAMPLE_RATE / 2,
        "power": 2
    }