import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
from torchvision import models
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
import pathlib
import numpy as np

#Hyperparameters
lr=0.005
weight_decay=0.0001
num_epochs = 100

#Define selectedv model to be trained
model = models.inception_v3(pretrained=True)

print(model)

for parameter in model.parameters():
    parameter.requires_grad = False

n_input=model.fc.in_features
print(n_input)
last_layer=nn.Linear(n_input, 10, bias=True)
model.fc=last_layer
model.AuxLogits.fc = nn.Linear(768, 10, bias=True)
for param in model.fc.parameters():
    param.requires_grad = True
for param in model.AuxLogits.fc.parameters():
    param.requires_grad = True
print(model)

model = model.to(device)

import torch.optim as optim

criterion_transfer = nn.CrossEntropyLoss()
params = list(model.fc.parameters()) + list(model.AuxLogits.fc.parameters())
optimizer_transfer = optim.Adam(params, lr=0.005, weight_decay=0.0001)

loaders_transfer={
    "train": train_loader,
    "test": test_loader
}