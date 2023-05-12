from scipy.fft import fft, fftfreq
from scipy.sparse import spdiags
import sklearn.decomposition
import sklearn.preprocessing
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
import warnings

class RollingNormalize:
    def __init__(self, window_size):
        self.window_size = window_size
        self.should_trim = True
        self.axis = 2

    def __call__(self, X):
        res = np.apply_along_axis(self.rolling_normalize, self.axis, X.numpy())
        X = torch.tensor(res, dtype=torch.float)
        return X
    
    def rolling_normalize(self, x):
        if len(x) < self.window_size:
            raise Exception(
                'Window size is greater than length (win: ' + str(self.window_size) + ', len: ' + str(len(x)) + ')'
                )
        windows = list(pd.Series(x).rolling(self.window_size))
        res_x = [self.normalize_by_window(x[idx], windows[idx]) for idx, _ in enumerate(x)]
        if self.should_trim:
            res_x = self.trim(res_x)
        return np.array(res_x)

    def normalize_by_window(self, x, w):
        std = np.std(w)
        if std <= 0.001:
            return np.average(w)
        return (x - np.average(w))* 1/std

    def trim(self, x):
        return x[self.window_size-1:]

class RPPGDetrend:
    def __init__(self, lam=10) -> None:
        self.lam = lam
        self.axis = 2

    def __call__(self, X):
        res = np.apply_along_axis(self.advanced_detrend, self.axis, X.numpy())
        X = torch.tensor(res, dtype=torch.float)
        return X

    def advanced_detrend(self, x):
        T = len(x)
        I = np.eye(T)
        A = (np.ones((T, 1)) * np.array([1, -2, 1])).T
        diags = np.array([0, 1, 2])
        D2 = spdiags(A, diags, T-2, T).toarray()
        z_stationary = (I - np.linalg.inv(I + self.lam**2 * D2.T @ D2)) @ x
        return z_stationary.tolist()#[0]

class SignalToPowerAndFreq:

    def __init__(
            self,
            sampling_rate,
            normalize=True, 
            freq_range=(30/60, 250/60)
            ):
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.freq_range = freq_range

    def __call__(self, X):
        res = np.apply_along_axis(self.peak_power_and_freq, 2, X.numpy())
        res = torch.tensor(res, dtype=torch.float)
        return res
    
    def rppg_fft_and_fftreq(self, x):
        freq = fftfreq(len(x), self.sampling_rate)
        power = abs(fft(x))
        norm = np.linalg.norm(power/np.max(power)) if self.normalize else 1 # 'max' to prevent errors with large numbers when taking norm.
        power /= norm
        valid_idxs = self.select_range(freq, self.freq_range[0], self.freq_range[1])
        power, freq = np.take(power, valid_idxs), np.take(freq, valid_idxs)
        return power, freq
    
    def peak_power_and_freq(self, x):
        power, freq = self.rppg_fft_and_fftreq(x)
        peak_idx = np.argmax(power)
        return power[peak_idx], freq[peak_idx]

    def select_range(self, t, start, end):
        idxs = [idx for idx, ti in enumerate(t) if start <= ti and ti <= end]
        return idxs

class MaxFreqChannelSelector:

    def __call__(self, X):
        res = np.array([self.freq_peak(x) for x in X.numpy()])
        res = torch.tensor(res, dtype=torch.float)
        return res
    
    def freq_peak(self, x):
        res = []
        for i in range(x.shape[2]):
            idx_max = np.argmax(x[:,0,i])
            res.append(x[idx_max,1,i])
        return np.array(res)

class RPPGICA:

    def __call__(self, X):
        res = np.array([self.rgb_ica(x) for x in X.numpy()])
        return torch.tensor(res, dtype=torch.float)
    
    def rgb_ica(self, x):
        warnings.filterwarnings("ignore")
        ica = sklearn.decomposition.FastICA(n_components=3, whiten=False) #whiten='unit-variance'
        if len(x) != 3:
                raise Exception('Number of channels must be exactly 3')
        res = []
        for i in range(x.shape[-1]):
            xi = x[:,:,i]
            #l = np.nan_to_num((channels), 0).astype(int)#.tolist() # For some really weird bug. There are some cases where ICA will say there are nan of inf values (even when there is not) and the only way to solve it is by converting to list and passing it as list, even though it hold the exact same values as the array. 
            sig_mat = ica.fit_transform(xi.T) # Don't need to transpose back because later transpose
            res.append(sig_mat)
        warnings.filterwarnings("always")
        return np.transpose(np.array(res), (2,1,0))

class HRPredict:
    def __init__(
            self,
            sampling_rate,
            freq_range=(30/60, 250/60)
            ):
        self.power_and_freq = SignalToPowerAndFreq(sampling_rate, freq_range=freq_range)
        self.selector = MaxFreqChannelSelector()
        self.predict = transforms.Compose([
            self.power_and_freq,
            self.selector
        ])

    def __call__(self, X):
        if X.dim() == 2:
            X = X.view(X.shape[0], 1, X.shape[1], 1)
            return self.predict(X)
        raise Exception('X dim is not 2.')
    
    def rppg_fft_and_fftreq(self, x):
        return self.power_and_freq.rppg_fft_and_fftreq(x)

class AverageModule:
    def __call__(self, X):
        return torch.tensor(np.average(X, axis=1), dtype=torch.float)
    
class MedianModule:
    def __call__(self, X):
        return torch.tensor(np.median(X, axis=1), dtype=torch.float)
        