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


# Creating a PyTorch class
# 480*480 ==> 10
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # Input shape= (256,3,480,480)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(480 * 480 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 10)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

modelA = torchvision.models.densenet121(pretrained=False,drop_rate=drop_rate)
modelB = AE()
num_ftrs = modelA.classifier.in_features
modelA.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    nn.Linear(500, 10)
)


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(20, 10)

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

model = MyEnsemble(modelA, modelB)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-4)

model = model.to(device)