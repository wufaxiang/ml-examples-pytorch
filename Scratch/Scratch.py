#!/usr/bin/env python
# import matplotlib.pyplot as plt
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

logging.basicConfig(filename='scratch.log', level=logging.INFO)

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
valid_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

'''
def plot_image(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image, cmap='gray')
    plt.show()

# %% Plot the sample image
sample_data = next(iter(train_loader))
plot_image(sample_data[0][1])
plot_image(sample_data[0][2])
'''


# %% Net model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# model_net = Net()
model_net = torch.load('model/scratch.ml')
if is_cuda:
    model_net = model_net.cuda()

optimizer = optim.SGD(model_net.parameters(), lr=0.01)

dataloaders = {'training': train_loader, 'validation': valid_loader}


# %% Train the model
def fit(model, data_lds, volatile=False, num_epoch=2):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epoch):
        for phase in ['training', 'validation']:
            data_loader = data_lds[phase]
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

            logging.info(
                '---------- EPOCH {:0>2d} - {:^10} loss: [ {:6f} ]  accuracy:[ {:5d}/{:5d}={:.4%} ] ----------'.format(
                    epoch,
                    phase,
                    loss_val,
                    running_correct,
                    data_size,
                    accuracy_val))
            if phase == 'validation' and accuracy_val > best_acc:
                best_acc = accuracy_val
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


# %% Run
train_loss, train_accuracy = [], []
valid_loss, valid_accuracy = [], []

model_ft = fit(model_net, dataloaders, num_epoch=50)
torch.save(model_ft.cpu(), 'model/scratch.ml')
logging.info('------------------------- TRAIN END _______________________________')
'''
#%% Plot image
plt.plot(range(1, len(train_loss)+1), train_loss, 'bo', label='training loss')
plt.plot(range(1, len(valid_loss)+1), valid_loss, 'r', label='validation loss')
plt.legend()
plt.show()

plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'bo', label='train accuracy')
plt.plot(range(1, len(valid_accuracy)+1), valid_accuracy, 'r', label='val accuracy')
plt.legend()
plt.show()
'''

