import os
from lib.models.rppg_transforms import RollingNormalize, RPPGDetrend, RPPGICA, HRPredict, AverageModule, SignalToPowerAndFreq, MaxFreqChannelSelector, MedianModule
from torchvision import transforms
from lib.models.resnet import RPPGResNet
from lib.models.rnn import RPPGRNN
from lib.models.rppgnn import RPPGNN
import torch

class ProjectModelHandler:
    def __init__(self, models_path, sampling_rate):
        self.models_path = models_path
        self.sampling_rate = sampling_rate
        

    def get_model_path(self, model_name):
        return os.path.join(self.models_path, model_name)
    
    def load_model(self, model_name, pretrained=True, eval_state=False):
        if model_name == 'ica_avg':
            return transforms.Compose([
                RPPGICA(),
                SignalToPowerAndFreq(self.sampling_rate),
                MaxFreqChannelSelector(),
                AverageModule()
                ])
        
        if model_name == 'ica_med':
            return transforms.Compose([
                RPPGICA(),
                SignalToPowerAndFreq(self.sampling_rate),
                MaxFreqChannelSelector(),
                MedianModule()
                ])
        
        if model_name == 'nn':
            model = RPPGNN(self.sampling_rate)
            if pretrained:
                state_dict = torch.load(self.get_model_path(model_name))
                model.load_state_dict(state_dict)
            if eval_state:
                model.eval()
            return model
        
        elif model_name == 'resnet':
            model = RPPGResNet()
            if pretrained:
                state_dict = torch.load(self.get_model_path(model_name))
                model.load_state_dict(state_dict)
            if eval_state:
                model.eval()
            return model
        
        elif model_name == 'tresnet':
            model = RPPGResNet()
            if pretrained:
                state_dict = torch.load(self.get_model_path(model_name))
                model.load_state_dict(state_dict)
            if eval_state:
                model.eval()
            return model
        
        elif model_name == 'rnn':
            model = RPPGRNN(2)
            if pretrained:
                state_dict = torch.load(self.get_model_path(model_name))
                model.load_state_dict(state_dict)
            if eval_state:
                model.eval()
            return model
        
        elif model_name == 'trnn':
            model = RPPGResNet()
            if pretrained:
                state_dict = torch.load(self.get_model_path(model_name))
                model.load_state_dict(state_dict)
            if eval_state:
                model.eval()
            return model
