import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vmodels

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