import torch 
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import numpy as np
from utils.preprocessing import Preprocessing
import matplotlib.pyplot as plt
from collections import Counter
class Capture_128(Dataset):
    def __init__(self, root, isTrain=True, transform=None):
        self.root = root
        self.data_frame = pd.read_feather(root)
        self.transform = transform
        self.isTrain = isTrain
        self.samples, self.labels = self._get_data()

    def _get_data(self):
        samples = np.array(self.data_frame.iloc[:,1:-1])
        labels = np.array(self.data_frame.iloc[:,-1])
        if self.isTrain:
            pre_processing = Preprocessing()
            samples, labels = pre_processing.fit_transform(samples, labels) # Make balanced dataset
        print(samples.shape)
        for col in range(samples.shape[1]):
            x_i = samples[:, col]
            x_max = np.max(x_i)
            x_min = np.min(x_i)
            samples[:,col] = (x_i-x_min)/(x_max-x_min)
        return samples, labels 
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return torch.Tensor(self.samples[index]), int(self.labels[index])
    
if __name__=="__main__":
    dataset = Capture_128('dataset/Capture_train_128.feather')
    freqs = Counter(np.sort(dataset.labels))
