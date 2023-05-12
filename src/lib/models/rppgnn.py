import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from .rppg_transforms import RPPGICA, SignalToPowerAndFreq
from torchvision import transforms

class RPPGNN(nn.Module):
    def __init__(self, sampling_rate):
        super(RPPGNN, self).__init__()
        n_regions = 25
        self.p = transforms.Compose([SignalToPowerAndFreq(sampling_rate)]) # RPPGICA()
        self.m = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_regions*2*3, n_regions*2*3),
            nn.ReLU(),
            nn.Linear(n_regions*2*3, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 1)
        )


    def forward(self, x):
        x = self.p(x)
        x = self.m(x)
        return x