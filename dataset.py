import torch 
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import numpy as np
class Capture_128(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.data_frame = pd.read_feather(root)
        self.samples, self.labels = self._get_data()
        self.transform = transform
        self.train = train
    def _get_data(self):
        samples = self.data_frame.iloc[:,1:-1]
        labels = self.data_frame.iloc[:,-1]
        for col in range(samples.shape[1]):
            feature = samples.iloc[:,col]
            samples.iloc[:,col] = (feature-min(feature))/(max(feature)-min(feature))
        return np.array(samples), np.array(labels) 

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return torch.Tensor(self.samples[index]), int(self.labels[index])