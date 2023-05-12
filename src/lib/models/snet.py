import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

class StNet(nn.Module):
    def __init__(self):
        super().__init__()
        n_region=36
        c = 3*300*n_region
        v = 12*int((300-6)/4)*int((n_region-6)/4)
        self.conv1 = nn.Conv2d(3, 6, (5,5)) # 6*296*12
        #self.pool1 =nn.MaxPool2d(2) # 6*148*6
        self.conv2 = nn.Conv2d(6, 12, (3,3)) # 12*146*4
        self.pool2 = nn.MaxPool2d(4) # 12*73*2
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(v, v)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(v, v)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(v, 1)
        

    def forward(self, x):
        x = self.conv1(x)
        #x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x