import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

mapping = {0: 'NEUTROPHIL', 1: 'EOSINOPHIL', 2: 'MONOCYTE', 3: 'LYMPHOCYTE'}
print("Loading model")

#names of the class and the class variables should match with those in model.py
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = self.conv(3,32)
        self.CompoundLayer1 = CompoundLayer(32)
        self.CompoundLayer2 = CompoundLayer(64)
        self.CompoundLayer3 = CompoundLayer(128)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 4)

    def conv(self, in_, out_): 
        return nn.Sequential(
            nn.Conv2d(in_, out_, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_))
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.CompoundLayer1(x)
        x = self.CompoundLayer2(x)
        x = self.CompoundLayer3(x)
        x = self.avgpool(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
# ----------------------------------------------------------------------------------------#
class CompoundLayer(nn.Module):
    def __init__(self, in_):
        super().__init__()
        self.layer1 = self.cone(in_, in_*2)
        self.layer2 = self.cone(in_*2, in_*2)
        self.layer2_1 = self.conv(in_*2, in_*2)
        
        self.layer1_1x1 = self.one_by_one(in_, in_*2)

    def one_by_one(self, in_, out_):
        return nn.Sequential(
            nn.Conv2d(in_, out_, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_)
        )
    def conv(self, in_, out_):
        return nn.Sequential(
            nn.Conv2d(in_, out_, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_))
    
    def cone(self, in_, out_): 
        mid = round(in_ * 3/2)
        return nn.Sequential(
            nn.Conv2d(in_, mid, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, out_, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_))
    def forward(self, x):
        x = self.layer1(x) + self.layer1_1x1(x)
        x = self.layer2(x) + x
        x = self.layer2_1(x)
        return x


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


net = Model()


model = sys.argv[1]
picture = sys.argv[2]

img = cv2.resize(cv2.imread(picture), (64,64))
image = np.asarray([img])

mean = tuple((image.mean(axis=(0,1,2))/255).round(4))
std = tuple((image.std(axis=(0,1,2))/255).round(4))

data = BloodDataset(image, [0], transform=
    transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
net.load_state_dict(torch.load(model))
print("Model Loaded")
net.eval()

input_image_loader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=5)
# input_image_loader = input_image_loader.view(-1, 3, 32, 128).cuda()
for batch_idx, (data, targets) in enumerate(input_image_loader):
        data = Variable(data)
        outputs = net(data)
        prediction = outputs.data.numpy().argmax()
        break

print(mapping[prediction])