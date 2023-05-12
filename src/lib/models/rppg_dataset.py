import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class RPPGDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_path = self.data_paths['x_path'].iloc[idx]
        y_path = self.data_paths['y_path'].iloc[idx]

        x = torch.tensor(np.load(x_path))
        y = torch.tensor(np.load(y_path))

        sample = x, y

        if self.transform:
            x = self.transform(x)

        return sample