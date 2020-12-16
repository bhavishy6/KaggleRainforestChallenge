import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import glob
import librosa
import librosa.display
import soundfile as sf
from scipy.interpolate import interp1d
import IPython.display as ipd

path = "data/train/00ad36516.flac"

waveform, sample_rate = torchaudio.load(path)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())
#plt.show()