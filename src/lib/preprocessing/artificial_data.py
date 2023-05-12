import numpy as np
import random
from .processing import min_max_norm_0_1
import torch
import torch.nn.functional as F

def get_random_heart_rate_signal_fn(omega1):
    M1 = random.uniform(0,1)
    phi = random.uniform(0, 2*np.pi)
    fn = lambda t: M1 * np.sin(omega1 * t + phi) + 0.5 * M1 * np.sin(2 * omega1 * t + phi)    
    return fn

def get_random_breathing_signal_fn(omega2):
    M2 = random.uniform(0,1)
    theta = random.uniform(0, 2*np.pi)
    fn = lambda t : M2 * np.sin(omega2 * t + theta)
    return fn

def get_random_illumination_fn(T):
    t1 = random.uniform(0, T)
    t2 = random.uniform(0, T)
    P1 = np.random.binomial(1, 0.5)
    P2 = np.random.binomial(1, 0.5)
    fn = lambda t : P1 * np.heaviside(t-t1, 1) + P2 * np.heaviside(t-t2, 1)
    return fn

def get_random_signal_rppg_sample(omega1_range, omega2_range, t, n_regions, t_offset_fn, noise_fn, rgb_intensities = np.array([1,1,1]), with_breathing=True, with_ilm=True):
    T = len(t)
    omega1 = np.random.uniform(omega1_range[0], omega1_range[1]) * 2*np.pi
    omega2 = np.random.uniform(omega2_range[0], omega2_range[1]) * 2*np.pi
    s_hr = get_random_heart_rate_signal_fn(omega1)
    s_b =  get_random_breathing_signal_fn(omega2) if with_breathing else lambda t : 0
    ilm = get_random_illumination_fn(T) if with_ilm else lambda t : 0
    s = lambda t : s_hr(t) + s_b(t) + ilm(t)
    x = (np.array([
        s(t + t_offset_fn())
        for _
        in range(n_regions)
        ])).reshape(n_regions,T,1)
    x = torch.tensor(x * rgb_intensities.reshape(1,3))
    x = torch.permute(x, (2, 1, 0)).numpy() + noise_fn()
    x = np.apply_along_axis(min_max_norm_0_1, 1, x)
    y = min_max_norm_0_1(s_hr(t))
    #print(x.shape)
    #print(x)
    #print(torch.tensor(x, dtype=torch.float).int())
    #print(torch.tensor(x, dtype=torch.float).int().shape)

    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

def sample_noise_from_data(X, t_size):
    idxs = np.random.randint(0, t_size, size=t_size)
    X = torch.permute(X, (0,2,1,3))
    X = torch.flatten(X, 0, 1)
    values = X[idxs]
    values = torch.permute(values, (1,0,2))
    return values