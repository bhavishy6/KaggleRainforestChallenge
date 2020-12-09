import torch
import torchaudio
import matplotlib.pyplot as plt

path = "data/train/00ad36516.flac"

waveform, sample_rate = torchaudio.load(path)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())
#plt.show()