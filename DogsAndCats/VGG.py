#!/usr/bin/env python
import os
import time
import logging
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder

logging.basicConfig(filename='dogs_cats.log', level=logging.INFO)
logging.info('---------------------- Train Start ----------------------')
path = '/home/faxiang/data/DogsAndCats/'

# Read all the files into our folder
files = glob(os.path.join(path, '*/*.jpg'))


def create_valid_set():
    size_images = len(files)

    # Create a shuffle to create valid data set
    shuffle = np.random.permutation(size_images)

    os.mkdir(os.path.join(path, 'valid'))

    for t in ['train', 'valid']:
        for folder in ['dog', 'cat/']:
            os.mkdir(os.path.join(path, t, folder))

    for i in shuffle[:2000]:
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'valid', folder, image))

    for i in shuffle[2000:]:
        folder = files[i].split('/')[-1].split('.')[0]
        image = files[i].split('/')[-1]
        os.rename(files[i], os.path.join(path, 'train', folder, image))


# create_valid_set()

def load_data():
    simple_transform = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train = ImageFolder(path + 'train/', simple_transform)
    valid = ImageFolder(path + 'valid/', simple_transform)

    train_data_gen = DataLoader(train, shuffle=True, batch_size=64, num_workers=3)
    valid_data_gen = DataLoader(valid, batch_size=64, num_workers=3)

    return train_data_gen, valid_data_gen


train_data, valid_data = load_data()
dataset_sizes = {'train': len(train_data.dataset), 'valid': len(valid_data.dataset)}
dataloaders = {'train': train_data, 'valid': valid_data}


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += float(loss.data.item())
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Create a network
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)

# model_ft = models.vgg16(pretrained=True)
# model_ft.classifier[6].out_features = 2
#for param in model_ft.features.parameters():
#    param.requires_grad = False

model_ft = torch.load('model/vgg.ml')

if torch.cuda.is_available():
    model_ft = model_ft.cuda()

# Loss and Optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.5)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)
torch.save(model_ft.cpu(), 'model/vgg.ml')
logging.info('---------------------- Train End ----------------------')
