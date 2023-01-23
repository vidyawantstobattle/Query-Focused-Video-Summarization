import pandas as pd
import torch
from torch import nn
import time
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np

def train(model, n_epochs, x):
    
    model = model.cuda()
    model.train()
    
    error = nn.MSELoss()

    optimizer = optim.Adam(model.parameters())
    
#     params = {'batch_size': 10,
#           'shuffle': True}
    
#     train_data = torch.utils.data.DataLoader(x, **params)

    for epoch in range(n_epochs):
        
        running_loss = 0.0
        start_time = time.time()  
        total_train_loss = 0
        
        for sample_batched in x:
        
            sample_batched = sample_batched.cuda()
            sample_batched = sample_batched.unsqueeze(0)

            optimizer.zero_grad()
            
            output, enc_out = model(sample_batched)
            
            loss = error(output, sample_batched)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() 
        
        print(f'epoch {epoch} \t Loss: {running_loss/len(x):.4g}')
        
    return model