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

#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)


# CNN Network

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((n-f+2P)/s) +1

        # Input shape= (256,3,1024,768)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,1024,768)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,1024,768)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,1024,768)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (256,12,512,384)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (256,20,512,384)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,512,384)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,512,384)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,512,384)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,512,384)

        self.fc = nn.Linear(in_features=512 * 384 * 32, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

        # Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,512,364)

        output = output.view(-1, 32 * 512 * 384)

        output = self.fc(output)
        output = self.softmax(output)
        return output

model=ConvNet(num_classes=10).to(device)

