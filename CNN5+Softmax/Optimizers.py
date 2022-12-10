import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

#Hyperparameters
num_epochs=55
lr=0.001
weight_decay=0.0001


#Optmizer and loss function
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_function = nn.CrossEntropyLoss()

#Calculating the size of training and testing images
train_count = len(glob.glob(train_path+'/**/*.jpg'))
test_count = len(glob.glob(test_path+'/**/*.jpg'))

print(train_count, test_count)