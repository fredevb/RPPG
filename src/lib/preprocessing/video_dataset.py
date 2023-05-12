import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class VideoRPPGDataset(Dataset):

    def __init__(self, csv_file, root_dir, ignore_y = False, transform=None):
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.ingore_y = ignore_y
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Some logic

        times = None
        images = None

        x = images, times
        y = None

        sample = x, y
        
        if self.transform:
            sample = self.transform(sample)

        return sample