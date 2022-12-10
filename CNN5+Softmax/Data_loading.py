

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

#Hyperparameter
batch_size = 32


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Data Transformation
training_transformer=transforms.Compose([
    transforms.Resize((1024,768)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,])
])
testing_transformer=transforms.Compose([
    transforms.Resize((1024,768)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,])
])



#Dataloader

#Path for training and testing directory
train_path='/home/ubuntu/DeepLearningData/DataChar/SEMClassify/TrainDataFull'
test_path='/home/ubuntu/DeepLearningData/DataChar/SEMClassify/TestDataFull'

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=training_transformer),
    batch_size=batch_size, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=testing_transformer),
    batch_size=batch_size, shuffle=True
)
