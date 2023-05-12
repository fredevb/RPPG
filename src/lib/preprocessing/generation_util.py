from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import random
import torch


# Using linear interpolation.
# In the future, it could be interesting to see if spline interpolation performs better.

def iter_to_rppg_data(video_data_loader, extractor, sampling_rate, t_size):
    for x, y in video_data_loader:
        images, times = x
        values = np.array([extractor.extract_regions_average(image) for image in images if not None])

        # Further divide the samples into smaller samples.
        n_images = len(images)
        t_samples = [
            np.arange(i * t_size * sampling_rate, (i+1) * t_size* sampling_rate, sampling_rate)
            for i in range(int(n_images / t_size))
        ]

        for ts in t_samples:
            yield np.interp(ts, times, values), y

# array format shape saved (n_roi, T, n_c + 1), where +1 is from y.
def save_rppg_sample(sample, x_path, y_path):
    x, y = sample
    np.save(x_path, x)
    np.save(y_path, y)

def save_rppg_dataset(
        iter_data, csv_path,
        data_root, 
        n_sample_upper_bound=np.inf, 
        x_name_fn = lambda i: 'sample_x' + str(i) + '.npy',
        y_name_fn = lambda i: 'sample_y' + str(i) + '.npy'
        ):
    x_paths = []
    y_paths = []
    for idx, sample in enumerate(iter_data):
        if (idx >= n_sample_upper_bound):
            break
        x_path = os.path.join(data_root, x_name_fn(idx))
        y_path = os.path.join(data_root, y_name_fn(idx))
        x_paths.append(x_path)
        y_paths.append(y_path)
        save_rppg_sample(sample, x_path, y_path)
    pd.DataFrame({'x_path' : x_paths, 'y_path' : y_paths}).to_csv(csv_path, index=False)

def rppg_train_validation_test_split(csv_path, p_val, p_test, train_csv_path, validation_csv_path, test_csv_path):
    csv_data = pd.read_csv(csv_path)
    n_data = len(csv_data)
    t_split = int(n_data * p_test)
    v_split = t_split + int(n_data * p_val)
    test_csv_data = csv_data[:t_split]
    validation_csv_data = csv_data[t_split:v_split]
    train_csv_data = csv_data[v_split:]
    test_csv_data.to_csv(test_csv_path, index=False)
    validation_csv_data.to_csv(validation_csv_path, index=False)
    train_csv_data.to_csv(train_csv_path, index=False)