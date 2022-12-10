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


#Plots the train loss and train accuracy across all epochs
plt.plot(train_loss_list, label='loss')
plt.plot(train_acc_list, label='accuracy')
plt.legend()
plt.title('training loss and accuracy')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.show()
###########################################################


#Plots the train loss and train accuracy across all epochs
plt.plot(test_loss_list, label='loss')
plt.plot(test_acc_list, label='accuracy')
plt.legend()
plt.title('testing loss and accuracy')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.show()
###########################################################


#Plots the values of train accuracy between 0.75-->1
plt.plot(train_acc_list, label='accuracy')
plt.legend()
plt.ylim(0.75,1)
plt.title('train accuracy')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.show()
###########################################################


#Plots the values of test accuracy between 0.75-->1
plt.plot(test_acc_list, label='accuracy')
plt.legend()
plt.ylim(0.75,1)
plt.title('test accuracy')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.show()
###########################################################
