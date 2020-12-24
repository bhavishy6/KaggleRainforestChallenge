import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as vmodels
import torchaudio.transforms as transforms
import glob
from tqdm.notebook import tqdm

class MelSpecDatasetV2(Dataset):
    def __init__(self):
        self.files = glob.glob( 'data/train/*.flac' )
        self.mel_specs = []
        
        for file in tqdm(self.files):
            waveform, sample_rate = torchaudio.load(file)
            print (waveform.shape)
            print (sample_rate)
        
        #sample_rate = int(self.source.iloc[0].sample_rate)
        #mel_trans = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=224)

        
    def __getitem__(self, idx):
        return self.mel_specs[idx]
    def __len__(self):
        return len(self.mel_specs)

class RainforestNet(nn.Module):
    def __init__(self):
        super(RainforestNet, self).__init__()
        self.resnet = vmodels.resnet50(pretrained=True)
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 225)
        self.fc3 = nn.Linear(225, 24)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.fc2(x)
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class RainforestNetV2(nn.Module):
    def __init__(self):
        super(RainforestNetV2, self).__init__()
        self.resnet = vmodels.resnet50(pretrained=True)
        self.fc1 = nn.Linear(1000, 500)
        self.d1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(500, 225)
        self.d2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(225, 24)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.sigmoid(self.fc3(x))

        return x

def reshape_mel_spec(mel_spec, max_len):
    diff = max_len - mel_spec.shape[2]
    if diff > 0:
        # pad
        s = int(diff/2) 
        e = diff - s
        mel_spec = F.pad(input=mel_spec, pad=(s, e, 0, 0, 0, 0), mode='constant', value=0)
    else:
        # trim
        mel_spec = mel_spec.narrow(2, diff*-1, max_len)

    mel_spec = mel_spec.repeat(3, 1, 1)
    return mel_spec