import pandas as pd
import torch
from torch import nn
import time
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,300)),
            nn.Linear(300, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, 100),
        )
        
        self.decode = nn.Sequential(
            
            nn.Linear(100, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 300),
            Interpolate(size = (8,300), mode='bilinear'),
        )
        
    def forward(self, x):
        
        x = self.encode(x)
        y = x
        
        x = self.decode(x)
        x = x.squeeze(0)
        
        return x, y

class Autoencoder_glove(nn.Module):

    def __init__(self):
        super(Autoencoder_glove, self).__init__()
        
        self.encode = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,200)),
            nn.Linear(200, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, 100),
        )
        
        self.decode = nn.Sequential(
            
            nn.Linear(100, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 200),
            Interpolate(size = (8,200), mode='bilinear'),
        )
        
    def forward(self, x):
        
        x = self.encode(x)
        y = x
        
        x = self.decode(x)
        x = x.squeeze(0)
        
        return x, y