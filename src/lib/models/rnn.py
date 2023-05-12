import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

class RPPGRNN(nn.Module):
    def __init__(self, n_divisions):
        super(RPPGRNN, self).__init__()
        self.n_divisions = n_divisions
        r = torchvision.models.resnet18(weights=None)
        n_in = r.fc.in_features
        self.resnet = nn.Sequential(*(list(r.children())[:-1]))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(n_in, 1024)
        self.rnn = nn.RNN(1024, 1024, 1)
        self.fc2 = nn.Linear(1024*n_divisions, 1)

    def forward(self, x):
        division_size = int(x.shape[2]/self.n_divisions)
        res = []
        for idx in range(len(x)):
            x_divs = [x[idx:idx+1,:,division_size*i:division_size*(i+1),:] for i in range(self.n_divisions)]
            output = [self.fc1(self.flat(self.resnet(x_div))) for x_div in x_divs]
            output = torch.flatten(torch.stack(output),1)
            res.append(output)

        x = torch.stack(res)
        x, h_n = self.rnn(x)
        x = self.flat(x)
        x = self.fc2(x)
        return x