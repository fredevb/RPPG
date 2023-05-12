from lib.preprocessing.artificial_data import get_random_signal_rppg_sample, sample_noise_from_data
from lib.preprocessing.generation_util import save_rppg_dataset, rppg_train_validation_test_split
from project_handlers.project_info import ProjectInfo
from project_handlers.project_data_handler import ProjectDataHandler
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import matplotlib.pylab as plt
import random

info = ProjectInfo()
t_size = info.t_size
data_root = info.data_root
sampling_rate = info.sampling_rate

# Parameters
extractor = None
dataset_name = 'artificial'
n_sample = 4000
sampling_dataset_name = None#'sampling'
n_regions = 25
with_breathing = True
with_ilm = True
t_offset_fn = lambda : 0#np.random.normal(0, 0.06,1)
noise_fn = lambda : (np.random.normal(0,0.1,(3, t_size, 1)).reshape((3, t_size, 1)) * np.ones((1,1,n_regions))) #+ np.random.normal(0,0.1,(3, t_size, n_regions))
rbg_intensities = np.array([1,0.9,0.8])

data_handler = ProjectDataHandler(info.data_root)

t = np.arange(0,t_size)*sampling_rate

sampling_noise = lambda : 0
if sampling_dataset_name is not None:
    sampling_dataset = data_handler.load_data(sampling_dataset_name)
    print('Using sampling')
    sampling_noise = lambda : sample_noise_from_data(next(iter(DataLoader(sampling_dataset, batch_size=10, shuffle=True)))[0], t_size).numpy()
    print(sampling_noise().shape)
#print(noise_fn().shape)
total_noise_fn = lambda : noise_fn() + sampling_noise()

fs = 1/sampling_rate
omega1_range = (0.7, 3.9)#(30/60*2*np.pi, 230/60*2*np.pi)
omega2_range = (0.08, 0.33)#(30/60*2*np.pi, 60/60*2*np.pi)


def data_iter(n_sample):
    for _ in range(n_sample):
        x, y = get_random_signal_rppg_sample(omega1_range, omega2_range, t, n_regions, t_offset_fn, total_noise_fn, rgb_intensities=rbg_intensities, with_breathing=with_breathing, with_ilm=with_ilm)
        yield x, y

csv_path = data_handler.get_data_csv_path(dataset_name)
dataset_root = data_handler.get_dataset_root(dataset_name)

save_rppg_dataset(
    data_iter(n_sample),
    csv_path,
    dataset_root,
    )

train_csv_path, validation_csv_path, test_csv_path = data_handler.get_train_validation_test_csv_paths(dataset_name)
rppg_train_validation_test_split(csv_path, 0.01, 0.01, train_csv_path, validation_csv_path, test_csv_path)

print('Created Artificial Dataset.')