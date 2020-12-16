# RainforestChallenge

## **Dataset**
https://www.kaggle.com/c/rfcx-species-audio-detection/data

Dataset was downloaded and extracted to a directory named 'data' at the root of the project. It is ignored by git. 

## **pip packages**

### **jupyterlab**
    pip install jupyterlab

### **pytorch**
    pip install torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

### **matplotlib**
    pip install matplotlib

### **numpy (correct version)**

Windows is having an issue importing torch with numpy 1.19.4, so uninstall numpy and install v1.19.3.

    pip uninstall numpy
    pip install numpy==1.19.3

### **SoundFile (audio i/o backend)**
    pip install SoundFile

### **pandas**
    pip install pandas
    
### **tqdm**
    pip install tqdm
    
### **librosa**
    pip install librosa
    