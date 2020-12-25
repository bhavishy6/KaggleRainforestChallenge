import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as vmodels
import torchaudio.transforms as transforms
import torchvision.transforms as vtransforms
import glob
import pandas as pd
from tqdm.notebook import tqdm

class Params():
    # train
    EPOCHS = 30
    TRAIN_BATCH_SIZE = 10
    LR = 0.0001
    
    ## Mel Spec
    
    # make both 0 for no noise, make both the same for no randomness, but RAND_NOISE_LOWER as non tp, non fp value
    RAND_NOISE_UPPER = 0.1
    RAND_NOISE_LOWER = 0.05

    N_MELS = 224
    
    SAMPLE_RATE = 48000
    N_MELS = 224
    N_FFT = 5000
    F_MIN = 90.0
    F_MAX = 14000.0

    def __to_dict__(self):
        return {key:value for key, value in Params.__dict__.items() if not key.startswith('__') and not callable(key)}

class MelSpecDatasetV2(Dataset):
    def __init__(self, params, train_test_split=None, stats=None, break_early=None):
        is_cuda = True and torch.cuda.is_available()
        cpu = torch.device('cpu')
        gpu = torch.device('cuda')
        device = gpu if is_cuda else cpu

        torch.cuda.empty_cache()
        
        self.params = params
        
        self.traint = pd.read_csv( 'data/train_tp.csv' )
        self.trainf = pd.read_csv( 'data/train_fp.csv' )
        self.files = glob.glob( 'data/train/*.flac' )
        self.ids = self.traint.recording_id.unique()
        self.mel_specs = []
        self.labels = []
        
        self.traint["val"] = 1
        self.trainf["val"] = 0
        
        self.train = pd.concat([self.traint, self.trainf])
        
        del self.traint
        del self.trainf
                
        mel_trans = transforms.MelSpectrogram(sample_rate=self.params.SAMPLE_RATE, n_mels=self.params.N_MELS, n_fft=self.params.N_FFT, f_min=self.params.F_MIN, f_max=self.params.F_MAX).to(device)
        normalize = vtransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
        
        f_maxs = []
        f_mins = []
        
        if train_test_split is not None:
            p, t = train_test_split
            train_len = int(len(self.ids)*p)
            if t:
                self.ids = self.ids[:train_len]
            else:
                self.ids = self.ids[train_len:]
                
        i=0
        for rec_id in tqdm(self.ids):
            path = "data/train/{}.flac".format(rec_id)
            waveform, sample_rate = torchaudio.load(path)
            waveform = waveform.to(device)
            mel_spec = mel_trans(waveform).repeat(3, 1, 1)
            #mel_spec = normalize(mel_spec)
            mel_spec = mel_spec.to(cpu)
                        
            del waveform
            torch.cuda.empty_cache()
                        
            self.mel_specs.append(mel_spec)
            
            subset = self.train[self.train['recording_id'] == rec_id]
            label = (self.params.RAND_NOISE_UPPER - self.params.RAND_NOISE_LOWER) * torch.rand(24) + self.params.RAND_NOISE_LOWER
            
            for idx in range(len(subset)):
                ex = subset.iloc[idx]
                
                f_maxs.append(ex.f_max)
                f_mins.append(ex.f_min)
                
                species_id = ex.species_id
                label[species_id] = ex.val
            self.labels.append(label)
            
            if break_early is not None and i > break_early:
                break
            i+=1
           
        if stats is not None:
            print (stats.describe(f_maxs))
            print (stats.describe(f_mins))
        
        del mel_trans
        torch.cuda.empty_cache()
    
    def get_recording_ids(self):
        return self.ids
    def __getitem__(self, idx):
        return (self.mel_specs[idx], self.labels[idx])
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
        self.resnet = vmodels.resnet34(pretrained=True)
        self.fc1 = nn.Linear(1000, 500)
        self.d1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(500, 225)
        self.d2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(225, 24)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        if self.training:
            x = self.d1(x)
        x = F.relu(self.fc2(x))
        if self.training:
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









# class MelSpecDataset(Dataset):
#     def __init__(self, source_file):
#         self.source = pd.read_csv(source_file)
        
#         mel_specs = []
#         wvs = []
#         sample_rate = int(self.source.iloc[0].sample_rate)
#         mel_trans = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=params.N_MELS).to(device)
        
#         for idx in tqdm(range(len(self.source))):
#             ex = self.source.iloc[idx]
#             waveform = ex.waveform

#             if isinstance(waveform, str): 
#                 wv = ','.join(ex.waveform.replace('[ ', '[').split())
#                 wv = np.array(ast.literal_eval(wv))
#                 waveform = torch.from_numpy(wv).view(1, -1).to(dtype=torch.float32)

#                 wvs.append(waveform)

#             sample_rate = int(ex.sample_rate)
            
#             waveform = waveform.to(device).view(1, 1, -1)
            
#             mel_spec = mel_trans(waveform)
#             mel_spec = reshape_mel_spec(mel_spec[0], params.MEL_SPEC_MAXLEN).to(cpu)
                        
#             mel_specs.append(mel_spec)

#         if 'mel_spec' in self.source:
#             self.source = self.source.assign(mel_spec=mel_specs)
#         else:
#             self.source.insert(4, "mel_spec", mel_specs, True)
            
#         # drop waveform data for now to save memory
#         self.source = self.source.drop(columns=['waveform'])
        
#         # uncomment to visualize waveform
#         #if len(wvs) > 0:
#         #    self.source = self.source.assign(waveform=wvs)
#     def get_waveform(self, idx):
#         ex = self.source.iloc[idx]
#         return (ex.waveform, ex.sample_rate)
#     def __getitem__(self, idx):
#         ex = self.source.iloc[idx]
#         return (ex.mel_spec, int(ex.species_id))
#     def __len__(self):
#         return len(self.source)