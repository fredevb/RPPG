import generation_util
from torch.utils.data import DataLoader
import os
import json
import pandas as pd

# Parameters
extractor = None
dataset = None
data_name = 'test'
n_sample_upper_bound = 1000000

# Instantiate data loader.
data_loader = DataLoader(dataset)

# Load project information.
project_root = '../..'
os.chdir(project_root)
with open('project_info.json', 'r') as f:
    info = json.load(f)

sampling_rate = info['sampling_rate']
t_size = info['t_size']
data_path = info['data_path']

# Compute
generation_util.save_rppg_dataset(
    generation_util.iter_to_rppg_data(data_loader, extractor, sampling_rate, t_size),
    os.path.join(data_path, data_name, 'data.csv'),
    os.path.join(data_path, data_name),
    n_sample_upper_bound
    )

print('Created Dataset.')