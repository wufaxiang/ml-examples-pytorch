#!/usr/bin/env python
# import matplotlib.pyplot as plt
import time
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
    print('Use GPU.')

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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


model = Net()
if is_cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)


# %% Train the model
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
        running_loss += F.nll_loss(output, target, reduction='sum').data.item()
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


# %% Run
train_loss, train_accuracy = [], []
valid_loss, valid_accuracy = [], []
best_acc = 0.0
since = time.time()
for epoch in range(1, 50):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
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

