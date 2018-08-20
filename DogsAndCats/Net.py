#!/usr/bin/env python
from glob import glob
import os
import numpy as np
# import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time

is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

simple_transform  = transforms.Compose([transforms.Resize((224,224))
                                       ,transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])
path = '/home/faxiang/data/DogsAndCats/'
train = ImageFolder(path + 'train/', simple_transform)
valid = ImageFolder(path + 'valid/', simple_transform)

print(train.class_to_idx)
print(train.classes)

train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, num_workers=3, shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=32, num_workers=3, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 2)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

model = Net()
if is_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def fit(epoch, model, data_loader, phase='training', volatile=False):
    data_size = len(data_loader.dataset)
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        running_loss += float(F.nll_loss(output, target, reduction='sum').data.item())
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss_val = running_loss / data_size
    accuracy_val = float(running_correct) / data_size

    print('---------- EPOCH {:0>2d} - {:^10} loss: [ {:6f} ]  accuracy:[ {:5d}/{:5d}={:.4%} ] ----------'.format(epoch,
                                                                                                                 phase,
                                                                                                                 loss_val,
                                                                                                                 running_correct,
                                                                                                                 data_size,
                                                                                                                 accuracy_val))

    return loss_val, accuracy_val

train_loss, train_accuracy = [], []
valid_loss, valid_accuracy = [], []
best_acc = 0.0
since = time.time()
for epoch in range(1, 20):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, valid_data_loader, phase='validation')
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    valid_loss.append(val_epoch_loss)
    valid_accuracy.append(val_epoch_accuracy)
    if val_epoch_accuracy > best_acc:
        best_acc = val_epoch_accuracy

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
