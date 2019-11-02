from multiprocessing import freeze_support

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd.variable import Variable
import pandas as pd
import cv2
import numpy as np

DIR = "../data/blood-cells/dataset2-master/dataset2-master/images/"
TEST = "../data/blood-cells/dataset2-master/dataset2-master/images/TEST/"
TRAIN = "../data/blood-cells/dataset2-master/dataset2-master/images/TRAIN/"
MODEL_STORE_PATH = '../model/'
EPOCHS = 20
batch_size = 24
test_size = 12
lr = 0.0002
mappings = dict(zip(['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE','LYMPHOCYTE'],list(range(0,4))))
is_gpu_available = torch.cuda.is_available()

def get_data(folder):
    images = []
    labels = []
    for subtype in os.listdir(folder):
        if not subtype.startswith('.'):
            #label can be 0,1,2,3
            label = mappings[subtype]
        for img_name in os.listdir(folder + subtype):
            images.append(cv2.resize(cv2.imread(folder + subtype + '/' + img_name), (64,64)))
            labels.append(label)
    return np.asarray(images), np.asarray(labels)

train_im, train_labels = get_data(TRAIN)
test_im, test_labels = get_data(TEST)

class BloodDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx]), self.labels[idx]
        return self.images[idx], self.labels[idx]

def normalizedTensor(mean, std):
    #transforms an image with values in the range [0,255] to [0,1]
    return transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(mean, std)])

#calculating a tuple of mean and standard deviation when counting elements along 3 axis
mean = tuple((train_im.mean(axis=(0,1,2))/255).round(4))
std = tuple((train_im.std(axis=(0,1,2))/255).round(4))
train = BloodDataset(train_im, train_labels, transform=normalizedTensor(mean,std))

mean = tuple((test_im.std(axis=(0,1,2))/255).round(4))
std = tuple((test_im.std(axis=(0,1,2))/255).round(4))
test = BloodDataset(test_im, test_labels, transform=normalizedTensor(mean,std))

trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test, batch_size=test_size, shuffle=False)

model = Model()
if is_gpu_available:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
loss_fn = nn.CrossEntropyLoss()

def checkpoint(epoch):
    torch.save(model.state_dict(), MODEL_STORE_PATH + str(epoch) + ".model.epoch")
    print(epoch, "model saved")

def test(epoch):
    model.eval()
    test_loss = 0
    counter = 0
    test_accuracy = 0

    for batch_id , (data,targets) in enumerate(testloader):
        if is_gpu_available:
            data, targets = Variable(data).cuda(), Variable(targets).cuda()

        outputs = model(data)
        loss = loss_fn(outputs, targets)

        test_loss += loss.data.cpu().numpy().round(5)
        counter += 1

        _, predicted = torch.max(outputs.data, 1)
        test_accuracy += (predicted == targets.data).sum()
        
        if batch_id%(batch_size//10) == 0:
            print('Test Epoch:', epoch, batch_id*len(data),'/',len(testloader.dataset),
                  '|\n Test Loss:',loss.data.cpu().numpy().round(5),
                  '|\n Accuracy:', (test_accuracy*100/(targets.data.size(0)*(batch_id + 1)))
                 )
    return test_loss/counter

def train(num_epoch):
    best_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        counter = 0
        train_accuracy = 0
        for batch_id, (data, targets) in enumerate(trainloader):
            if is_gpu_available:
                data, targets = Variable(data).cuda(), Variable(targets).cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            train_loss += loss.data.cpu().numpy().round(5)
            counter += 1
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data,1)
            train_accuracy += (predicted == targets.data).sum()
        test_acc = test(epoch)

        if test_acc > best_acc:
            checkpoint(epoch)
            best_acc = test_acc

        train_loss = train_loss/counter

        if batch_id%(batch_size//10) == 0:
            print('Train Epoch:', epoch, batch_id*len(data),'/',len(trainloader.dataset),
                  '|\n Train Loss:',train_loss,
                  '|\n Accuracy:', (train_accuracy*100/(targets.data.size(0)*(batch_id+1)))
                 )

if __name__ == '__main__':
    freeze_support()
    train(EPOCHS)