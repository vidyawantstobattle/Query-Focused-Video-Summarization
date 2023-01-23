import pandas as pd
import torch
from torch import nn
import time
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np

class encoder(nn.Module):

    def __init__(self):
        super(encoder, self).__init__()
        
        self.lstm = nn.LSTM(300, 300, num_layers = 1, batch_first=True)
        self.fc_lstm = torch.nn.Linear(300, 400)  
        
    def forward(self, x):
        out, (ht,ct) = self.lstm(x)
        
        ht = self.fc_lstm(ht)

        return out, ht, ct