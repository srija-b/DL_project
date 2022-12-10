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


from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def inception_train(n_epochs, loaders, model, optimizer, criterion, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    test_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        test_loss = 0.0
        train_accuracy = 0.0
        test_accuracy = 0.0
        correct_train = 0.0
        correct_test = 0.0
        total_test = 0.0
        total_train = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            data, target = data.to(device), target.to(device)
            ## find the loss and update the model parameters accordingly
            optimizer.zero_grad()
            outputs, aux_outputs = model(data)
            loss1 = criterion(outputs, target)
            loss2 = criterion(aux_outputs, target)
            loss = loss1 + 0.4 * loss2
            loss.backward()
            optimizer.step()
            ## record the average training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # Calculating train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += target.size(0)

            correct_train += (predicted == target).sum().item()
            train_accuracy = correct_train / total_train

            if batch_idx % 100 == 99:
                print('Epoch %d, Batch %d loss: %.6f' %
                      (epoch, batch_idx + 1, train_loss))
        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            data, target = data.to(device), target.to(device)
            ## update the average validation loss
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

            # Calculating test accuracy
            _, prediction = torch.max(outputs.data, 1)
            total_test += target.size(0)

            correct_test += (prediction == target).sum().item()
            test_accuracy = correct_test / total_test

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
            epoch,
            train_loss,
            test_loss
        ))

        ## TODO: save the model if validation loss has decreased
        if test_loss < test_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                test_loss_min,
                test_loss))
            test_loss_min = test_loss
        print('Epoch: ' + str(epoch) + '| Train Loss: ' + str(train_loss) + '| Train Accuracy: ' + str(
            train_accuracy) + '| Test Accuracy: ' + str(test_accuracy))
        # return trained model
    return model


# train the model
model_transfer = inception_train(num_epochs, loaders_transfer, model, optimizer_transfer, criterion_transfer, 'model_transfer.pt')