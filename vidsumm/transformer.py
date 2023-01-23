import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# Define the transform 
train_transform = transforms.Compose([
        transforms.Resize((224,224)),             # takes PIL image as input and outputs PIL image
        transforms.ToTensor(),              # takes PIL image as input and outputs torch.tensor
        transforms.Normalize(mean=[0.4280, 0.4106, 0.3589],  # takes tensor and outputs tensor
                             std=[0.2737, 0.2631, 0.2601]),  
    ])

valid_transform = transforms.Compose([ 
        transforms.Resize((224,224)),             
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4280, 0.4106, 0.3589],
                             std=[0.2737, 0.2631, 0.2601]), 
    ])

test_transform = transforms.Compose([
        transforms.Resize((224,224)),             
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4280, 0.4106, 0.3589],
                             std=[0.2737, 0.2631, 0.2601]), 
    ])