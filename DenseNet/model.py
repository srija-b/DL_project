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



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torchvision.models.densenet121(pretrained=False,drop_rate=drop_rate)
num_ftrs = model.classifier.in_features
print(num_ftrs)
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    nn.Linear(500, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)
model = model.to(device)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []


for epoch in range(epochs):
    print("Epoch", epoch + 1, "/", epochs)
    model.train()
    train_loss = 0
    train_acc = 0
    itr = 1
    tot_itr = len(train_loader)
    for samples, labels in tqdm.tqdm(train_loader, desc="Training", unit=" Iterations"):
        samples, labels = samples.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(samples)
        loss = criterion(output, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(labels)
        train_acc += torch.mean(correct.float())
        torch.cuda.empty_cache()
        itr += 1

    train_loss_list.append(train_loss / tot_itr)
    train_acc_list.append(train_acc.item() / tot_itr)
    print(' Total Loss: {:.4f}, Accuracy: {:.1f} %'.format(train_loss, train_acc / tot_itr * 100))

    model.eval()
    test_loss = 0
    test_acc = 0
    itr = 1
    tot_itr = len(test_loader)
    for samples, labels in tqdm.tqdm(test_loader, desc="Testing", unit=" Iterations"):
        with torch.no_grad():
            samples, labels = samples.to(device), labels.to(device)
            output = model(samples)
            loss = criterion(output, labels)
            test_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(labels)
            test_acc += torch.mean(correct.float())
            torch.cuda.empty_cache()
            itr += 1

    test_loss_list.append(test_loss / tot_itr)
    test_acc_list.append(test_acc.item() / tot_itr)
    print('-----------------------------> Test Loss: {:.4f}, Accuracy: {:.1f} %'.format(test_loss,
                                                                                              test_acc / tot_itr * 100))

