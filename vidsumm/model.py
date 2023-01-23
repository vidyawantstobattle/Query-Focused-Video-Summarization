import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import resnet34
from torchvision.models import densenet201
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from . import encoder
from . import SelfAttention
from . import LSTM_encoder

class VidSmodel(nn.Module):

    def __init__(self):
        super(VidSmodel, self).__init__()
        
        self.model = resnet34(pretrained='imagenet')
         
        self.model = resnet34(pretrained=True) 
        self.fc1 = torch.nn.Linear(512, 4)
        
        self.fc_text = torch.nn.Linear(300, 512)
        

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)     
    
    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)
    
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
        t1 = torch.mul(x, y)

        #Computes the second fully connected layer
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction
    

class VidSmodel2(nn.Module):

    def __init__(self):
        super(VidSmodel2, self).__init__()
        
        self.model = resnet34(pretrained='imagenet')
         
        self.model = resnet34(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        
        self.fc_text = torch.nn.Linear(300, 512)
        

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)     
    
    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)
    
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction


class VidSmodel2_DenseNet(nn.Module):

    def __init__(self):
        super(VidSmodel2_DenseNet, self).__init__()
        
        self.model = densenet201(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        self.fc = torch.nn.Linear(1920, 512)
        
        self.fc_text = torch.nn.Linear(300, 512)
        

    def forward(self, x, y):
        x = self.model.features(x)    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction


class VidSmodel2_glove(nn.Module):

    def __init__(self):
        super(VidSmodel2_glove, self).__init__()
        
        self.model = resnet34(pretrained='imagenet')
         
        self.model = resnet34(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        
        self.fc_text = torch.nn.Linear(200, 512)
        

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)     
    
    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)
    
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction
    
class VidSmodel2_glove_denseNet(nn.Module):

    def __init__(self):
        super(VidSmodel2_glove_denseNet, self).__init__()
        
        self.model = densenet201(pretrained=True)
 
        self.fc1 = torch.nn.Linear(1, 4)
        self.fc = torch.nn.Linear(1920, 512)
        
        self.fc_text = torch.nn.Linear(200, 512)
        

    def forward(self, x, y):
        x = self.model.features(x)    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)
    
        x = self.fc(x)
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction
    
class VidSmodel3(nn.Module):

    def __init__(self, enc_model):
        super(VidSmodel3, self).__init__()
        
        self.model = resnet34(pretrained='imagenet') 
        self.model = resnet34(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        
        self.enc = encoder.Autoencoder()
        self.enc = enc_model
        self.fc_text = torch.nn.Linear(100, 512)
        

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)     
    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)

        _, y = self.enc(y)
        
        y = y.squeeze(1)
    
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction
    

class VidSmodel4(nn.Module):

    def __init__(self, enc_model):
        super(VidSmodel4, self).__init__()
        
        self.model = resnet34(pretrained='imagenet') 
        self.model = resnet34(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        
        self.enc = encoder.Autoencoder()
        self.enc = enc_model
        self.fc_text = torch.nn.Linear(100, 512)
        
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)     
    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)

        _, y = self.enc(y)
        
        y = y.squeeze(1)
    
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        t1 = self.dropout(t1)
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction


class VidSmodel4_DenseNet(nn.Module):

    def __init__(self, enc_model):
        super(VidSmodel4_DenseNet, self).__init__()
        
        self.model = densenet201(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        self.fc = torch.nn.Linear(1920, 512)
        
        self.enc = encoder.Autoencoder()
        self.enc = enc_model
        self.fc_text = torch.nn.Linear(100, 512)
        
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, y):
        x = self.model.features(x)    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        _, y = self.enc(y)
        
        y = y.squeeze(1)
    
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        t1 = self.dropout(t1)
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction


class VidSmodel5(nn.Module):

    def __init__(self,enc_model):
        super(VidSmodel5, self).__init__()
        
        self.model = resnet34(pretrained='imagenet') 
        self.model = resnet34(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        #self.fc = torch.nn.Linear(512*2, 512)
        
        self.enc = encoder.Autoencoder()
        self.enc = enc_model
        self.fc_text = torch.nn.Linear(100, 512)

        self.selfAtt = SelfAttention.SelfAttention(512,512,512,512)
        
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)     
    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)

        _, y = self.enc(y)
        
        y = y.squeeze(1)
    
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        y, y_weights, x, x_weights = self.selfAtt(y, x)
        #t1 = torch.mul(x, y)
        #t1 = torch.cat((y,x), 1)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        t1 = self.dropout(t1)
        #t1 = F.relu(self.fc(t1))
        
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction


class VidSmodel5_DenseNet(nn.Module):

    def __init__(self,enc_model):
        super(VidSmodel5_DenseNet, self).__init__()
        
        self.model = densenet201(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        self.fc = torch.nn.Linear(1920, 512)
        
        self.enc = encoder.Autoencoder()
        self.enc = enc_model
        self.fc_text = torch.nn.Linear(100, 512)

        self.selfAtt = SelfAttention.SelfAttention(512,512,512,512)
        
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, y):
        x = self.model.features(x)
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        _, y = self.enc(y)
        
        y = y.squeeze(1)
    
        y = F.relu(self.fc_text(y))
        
        #Combine x and y by element-wise multiplication. The output dimension is (1, 512).
#         t2 = torch.mul(x, y)

        y, y_weights, x, x_weights = self.selfAtt(y, x)
        #t1 = torch.mul(x, y)
        #t1 = torch.cat((y,x), 1)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        t1 = self.dropout(t1)
        #t1 = F.relu(self.fc(t1))
        
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction


class VidSmodel6(nn.Module):

    def __init__(self):
        super(VidSmodel6, self).__init__()
        
        self.model = resnet34(pretrained='imagenet') 
        self.model = resnet34(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        #self.fc = torch.nn.Linear(512*2, 512)
        
        #self.enc = encoder.Autoencoder()
        self.lstm_enc = LSTM_encoder.encoder()
        self.fc_text = torch.nn.Linear(400, 512)

        self.selfAtt = SelfAttention.SelfAttention(512,512,512,512)
        
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)     
    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)

        out, y, ct = self.lstm_enc(y)
        
        #y = y.sum(axis=0)
        y = y.squeeze(0)
        #y = self.dropout(y)

        y = F.relu(self.fc_text(y))

        y, y_weights, x, x_weights = self.selfAtt(y, x)
        #t1 = torch.mul(x, y)
        #t1 = torch.cat((y,x), 1)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        t1 = self.dropout(t1)
        #t1 = F.relu(self.fc(t1))
        
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction

class VidSmodel7(nn.Module):

    def __init__(self):
        super(VidSmodel7, self).__init__()
        
        self.model = resnet34(pretrained='imagenet') 
        self.model = resnet34(pretrained=True) 
        self.fc1 = torch.nn.Linear(1, 4)
        #self.fc = torch.nn.Linear(512*2, 512)
        
        self.lstm_enc = LSTM_encoder.encoder()
        self.fc_text = torch.nn.Linear(300, 512)
        
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)     
    
        x = F.avg_pool2d(x, 7)
        
        # reshape x
        x = x.view(x.size(0), -1)

        out, y, ct = self.lstm_enc(y)
        
        #y = y.sum(axis=0)
        y = y.squeeze(0)
        #y = self.dropout(y)

        y = F.relu(self.fc_text(y))

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        t1 = cos(x, y)
        t1 = torch.reshape(t1,(t1.shape[0],1))

        #Computes the second fully connected layer
        t1 = self.dropout(t1)
        #t1 = F.relu(self.fc(t1))
        
        relevance_class_prediction = self.fc1(t1)
        
        return relevance_class_prediction