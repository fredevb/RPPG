from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from lib.models.rppg_transforms import HRPredict
import torch
from torchvision import transforms

def get_evaluation(y_pred, y_true, bpm=False):
    if bpm:
        y_pred *= 60
        y_true *= 60
    error = y_pred - y_true
    mean = np.mean(error)
    std = np.std(error)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    #r = pearsonr(y_pred, y_true)
    ret = {'mean' : mean, 'std' : std, 'mse' : mse, 'mae' : mae}
    return ret

def get_results(project_info, data_handler, model_handler, dataset_names, model_names, transform=transforms.Compose([])):
    hr_predict = HRPredict(project_info.sampling_rate)
    ret = []
    for dataset_name in dataset_names:
        _, _, dataset = data_handler.load_train_validation_test(dataset_name)
        data_loader = DataLoader(dataset)
        for model_name in model_names:
            model = model_handler.load_model(model_name, pretrained=True, eval_state=True)
            yhat, y = zip(*[(model(transform(X.type(torch.float))), hr_predict(y))
                    for X, y 
                    in data_loader
                    ])
            yhat = torch.tensor(list(yhat)).numpy()
            y = torch.tensor(list(y)).numpy()
            ret.append((dataset_name, model_name, yhat, y))
    return ret

def get_evaluation_df(results):
    df = pd.DataFrame()
    for dataset_name, model_name, yhat, y in results:
        metrics = get_evaluation(yhat, y)
        row = pd.Series(metrics, name=dataset_name + ' & ' + model_name)
        df = df.append(row)
    return df