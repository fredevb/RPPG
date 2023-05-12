from project_handlers.project_info import ProjectInfo
from project_handlers.project_data_handler import ProjectDataHandler
from project_handlers.project_model_handler import ProjectModelHandler
import utils.plotting as plotting
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from lib.models.rppg_transforms import HRPredict
from lib.models.training import train_network
import torch
from torch import nn
from torchvision import transforms
from lib.models.rppg_transforms import RollingNormalize, RPPGDetrend

info = ProjectInfo()
t_size = info.t_size
data_root = info.data_root
model_root = info.model_root
sampling_rate = info.sampling_rate

model_name = 'resnet'
artificial_dataset_name = 'artificial'
real_dataset_name = 'mesh'
batch_size = 32
shuffle = False
learning_rate = 1e-4
momentum = 0.9
loss_fn = nn.L1Loss()
hr_predict = HRPredict(sampling_rate)
save_intermediate = True
save_final = True
transform = transforms.Compose([])


data_handler = ProjectDataHandler(data_root)
model_handler = ProjectModelHandler(model_root, sampling_rate)

model = model_handler.load_model(model_name, pretrained=False)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train(True)

if artificial_dataset_name is not None:
    artificial_dataset, validation_dataset, _ = data_handler.load_train_validation_test(artificial_dataset_name)
    artificial_data_loader = DataLoader(artificial_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_data = next(iter(DataLoader(validation_dataset, batch_size=10)))
    c, l, v = train_network(model, artificial_data_loader, loss_fn, optimizer, hr_predict=hr_predict, validation_set=validation_data, transform=transform)
    if save_intermediate:
        torch.save(model.state_dict(), model_handler.get_model_path('temp_' + model_name))
    plt.plot(c, l)
    plt.plot(c, v)
    plt.show()

if real_dataset_name is not None:
    real_dataset, validation_dataset, _ = data_handler.load_train_validation_test(real_dataset_name)
    real_data_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_data = next(iter(DataLoader(validation_dataset, batch_size=10)))
    c, l, v = train_network(model, real_data_loader, loss_fn, optimizer, hr_predict=hr_predict, validation_set=validation_data, transform=transform)
    torch.save(model.state_dict(), model_handler.get_model_path(model_name))
    plt.plot(c, l)
    plt.plot(c, v)
    plt.show()