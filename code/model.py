import torch
from torch import nn

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