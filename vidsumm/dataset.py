import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
from matplotlib import cm
import pandas as pd
import numpy as np
from. import transformer

class dataset(Dataset):

    def __init__(self, csv_file, root_dir, w2vmodel, transform):
        """
        Args:
            csv_file : Path to the csv file with annotations.
            root_dir : Directory with all the frames.
            transform : transform to be applied
                on a sample.
        """
        
        self.query_frame_train = pd.read_csv(csv_file)
        index = self.query_frame_train[self.query_frame_train['query']=='360 jeezy'].index
        index1 = self.query_frame_train[self.query_frame_train['query']=='barbie movies'].index
        index2 = self.query_frame_train[self.query_frame_train['query']=='arsenal vs watford'].index
        self.query_frame_train.drop(index, inplace = True)
        self.query_frame_train.drop(index1, inplace = True)
        self.query_frame_train.drop(index2, inplace = True)
        self.root_dir = root_dir
        self.w2vmodel = w2vmodel
        self.transform = transform

    def __len__(self):
        return len(self.query_frame_train) 

    

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,
                    self.query_frame_train.iloc[idx, 2].split("/")[len(self.query_frame_train.iloc[idx, 2].split("/"))-2],
                    self.query_frame_train.iloc[idx, 2].split("/")[len(self.query_frame_train.iloc[idx, 2].split("/"))-1])
        image = io.imread(img_name) 
        query = self.query_frame_train.iloc[idx, 0]
        score_annotations = self.query_frame_train.iloc[idx, 3:] 
        score_annotations = np.array([score_annotations])

        score_annotations = score_annotations.astype('float').reshape(-1, )
        
        query = query.lower()
        query = ' '.join(word for word in query.split(' ') if word in self.w2vmodel.vocab)
        words = query.split()
        SEQ_LENGTH = 8
        num_features = 300
        BATCH_SIZE = 1
        qdata = np.zeros((SEQ_LENGTH, num_features), dtype=np.float32)
        
        for j in range(SEQ_LENGTH):
            if j < len(words):
                qdata[j, :] = np.array(self.w2vmodel[str(words[j])])
        
        qdata = qdata.mean(axis=0)
        
        sample = {'image': image, 'query': qdata, 'score_annotations': score_annotations}

        #if self.transform:
        sample['image'] = self.transform(Image.fromarray(sample['image']))
        sample['score_annotations'] = torch.from_numpy(sample['score_annotations'])
        sample['query'] = torch.from_numpy(sample['query'])
        #print(sample['score_annotations'])
        return sample

class dataset_glove(Dataset):

    def __init__(self, csv_file, root_dir, glove, transform):
        """
        Args:
            csv_file : Path to the csv file with annotations.
            root_dir : Directory with all the frames.
            transform : transform to be applied
                on a sample.
        """
        
        self.query_frame_train = pd.read_csv(csv_file)
        index = self.query_frame_train[self.query_frame_train['query']=='360 jeezy'].index
        index1 = self.query_frame_train[self.query_frame_train['query']=='barbie movies'].index
        index2 = self.query_frame_train[self.query_frame_train['query']=='arsenal vs watford'].index
        self.query_frame_train.drop(index, inplace = True)
        self.query_frame_train.drop(index1, inplace = True)
        self.query_frame_train.drop(index2, inplace = True)
        self.root_dir = root_dir
        self.glove = glove
        self.transform = transform

    def __len__(self):
        return len(self.query_frame_train) 

    

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir,
                    self.query_frame_train.iloc[idx, 2].split("/")[len(self.query_frame_train.iloc[idx, 2].split("/"))-2],
                    self.query_frame_train.iloc[idx, 2].split("/")[len(self.query_frame_train.iloc[idx, 2].split("/"))-1])
        image = io.imread(img_name) 
        query = self.query_frame_train.iloc[idx, 0]
        score_annotations = self.query_frame_train.iloc[idx, 3:] 
        score_annotations = np.array([score_annotations])

        score_annotations = score_annotations.astype('float').reshape(-1, )
        
        query = query.lower()
        query = ' '.join(word for word in query.split(' ') if word in self.glove.vocab)
        words = query.split()
        SEQ_LENGTH = 8
        num_features = 200
        BATCH_SIZE = 1
        qdata = np.zeros((SEQ_LENGTH, num_features), dtype=np.float32)
        
        for j in range(SEQ_LENGTH):
            if j < len(words):
                qdata[j, :] = np.array(self.glove[str(words[j])])
        
        qdata = qdata.mean(axis=0)
        
        sample = {'image': image, 'query': qdata, 'score_annotations': score_annotations}

        #if self.transform:
        sample['image'] = self.transform(Image.fromarray(sample['image']))
        sample['score_annotations'] = torch.from_numpy(sample['score_annotations'])
        sample['query'] = torch.from_numpy(sample['query'])
        #print(sample['score_annotations'])
        return sample