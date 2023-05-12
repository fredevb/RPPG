import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

class RPPGResNet(nn.Module):
    def __init__(self):
        super(RPPGResNet, self).__init__()
        #r = nn.Sequential(
        #    nn.Flatten(),
        #    nn.Linear(3*4*200, 3*4*200),
        #    nn.ReLU(),
        #    nn.Linear(3*4*200, 3*4*200),
        #    nn.ReLU(),
        #    nn.Linear(3*4*200, 3*4*200),
        #    nn.ReLU(),
        #    nn.Linear(3*4*200, 1000),
        #    nn.ReLU()
        #)
        r = torchvision.models.resnet18(weights=None)
        n_in = r.fc.in_features
        self.resnet = nn.Sequential(*(list(r.children())[:-1]))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(n_in, 1000)
        self.fc2 = nn.Linear(1000, 1)
        

    def forward(self, x):
        #x = F.normalize(x)
        x = self.resnet(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x