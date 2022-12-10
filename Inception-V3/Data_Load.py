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


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Transforms
train_transform = transforms.Compose([
    transforms.Resize((1024,768)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #Std and mean values correspond to ImageNet images
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((1024,768)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],#Std and mean values correspond to ImageNet images
                         std=[0.229, 0.224, 0.225])
])

#Path for training and testing directory
train_path = '/home/ubuntu/DeepLearningData/DataChar/SEMClassify/TrainDataFull'
test_path = '/home/ubuntu/DeepLearningData/DataChar/SEMClassify/TestDataFull'
#DataLoader


train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=train_transform),
    batch_size=32, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=test_transform),
    batch_size=32, shuffle=True
)

#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)