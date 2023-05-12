from multi_basic_estimator import MultiBasicEstimator
import pickle
import numpy as np

class MultiBasicEstimatorRegressor:
    def __init__(self, regressor, n_regions, sampling_rate, window_size = 40, advanced_detrending=False, use_area = False):
        self.regressor = regressor
        self.n_regions = n_regions
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.advanced_detrending = advanced_detrending
        self.use_area = use_area
        self.multi_basic_estimators = MultiBasicEstimator(
            n_regions, sampling_rate,
            window_size=window_size,
            advanced_detrending=advanced_detrending)

    def fit(self, X, y):
        mbe_preds = self.multi_basic_estimators(X)
        print(mbe_preds.shape)
        print(y.shape)
        self.regressor.fit(mbe_preds, y)

    def __call__(self, X):
        return np.array([self.forward(x) for x in X])

    def forward(self, x):
        mbe_preds = self.multi_basic_estimators([x])
        prediction = self.regressor.predict(mbe_preds)
        return prediction
    

def mber_save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def mber_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)