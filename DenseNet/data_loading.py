import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import tqdm

#Transforms
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
train_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(normMean,
                         normStd)
])

test_transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(normMean,
                         normStd)
])

#HYPERPARAMETERS########
epochs = 100
drop_rate = 0
batch_size = 32
lr = 0.005
weight_decay = 1e-4
#######################


#Dataloader

#Path for training and testing directory

train_path='/home/ubuntu/DeepLearningData/DataChar/SEMClassify/TrainDataFull'
test_path='/home/ubuntu/DeepLearningData/DataChar/SEMClassify/TestDataFull'



train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=train_transform),
    batch_size=batch_size, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=test_transform),
    batch_size=batch_size, shuffle=True
)
