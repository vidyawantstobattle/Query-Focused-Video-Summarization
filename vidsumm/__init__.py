from six.moves import configparser
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gensim
import torch
from . import model
from . import encoder
from . import encoder_training
from . import train
import gensim.downloader as api

# Load the configuration
# if not 'VIDSUM_DATA_DIR' in os.environ:
#     os.environ['VIDSUM_DATA_DIR']='./data'
# config = configparser.SafeConfigParser(os.environ)
# print('Loaded config file from %s' % config.read('%s/config.ini' % os.path.dirname(__file__))[0])
def get_glove_function():
  glove_model = api.load('glove-twitter-200')
  return glove_model

def get_word2vec_function():

    # Set word2vec model
    print('Load word2vec model...')
#     if (os.path.isfile(config.get('paths', 'word2vec_file'))):
#         w2vmodel = gensim.models.Word2Vec.load(config.get('paths', 'word2vec_file'))
#     else:
    #w2vmodel = gensim.models.Word2Vec.load(config.get('paths', 'word2vec_smallfile'))
    w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/MyDrive/New_Code/New/GoogleNews-vectors-negative300.bin', binary=True)
    return w2vmodel

def train_model(w2vmodel):
    
    print('Loading Model...')
    model1 = model.VidSmodel()
    print('Model Loaded Successfully')
    
    print('Training Model...')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 50, 0.0001, w2vmodel)
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list 

def train_model2(w2vmodel):
    
    print('Loading Model2...')
    model1 = model.VidSmodel2()
    print('Model Loaded Successfully')
    
    print('Training Model...')
    trained_model = train.trainNet(model1, 10, 30, 0.0001, w2vmodel)
    print(' Model Trained Successfully')
    return trained_model

def train_model3(w2vmodel):
    
    print('Loading Model3...')
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    model1 = model.VidSmodel3(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 30, 0.0001, w2vmodel)
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model4(w2vmodel):
    
    print('Loading Model4...')
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    model1 = model.VidSmodel4(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 50, 0.0001, w2vmodel)
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model5(w2vmodel):
    
    print('Loading Model5....cosine similarity with 0.001l2 and step decay per 25 epochs')
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    model1 = model.VidSmodel5(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 35, 0.001, w2vmodel)
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model6(w2vmodel):
    
    print('Loading Model6....cosine similarity lstm enc with an extra fc layer')
    model1 = model.VidSmodel6()
    print('Model Loaded Successfully')
    
    print('Training Model...')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 25, 0.001, w2vmodel)
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model7(w2vmodel):
    
    print('Loading Model7....cosine similarity lstm enc')
    model1 = model.VidSmodel7()
    print('Model Loaded Successfully')
    
    print('Training Model...')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 30, 0.0001, w2vmodel)
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list


def train_encoder(x):
    
    print("Loading encoder model...")
    enc = encoder.Autoencoder()
    print('encoder Loaded Successfully')
    
    print("training Encoder")
    trained_enc = encoder_training.train(enc, 25, x)
    print(' Encoder Trained Successfully')
    return trained_enc

def train_encoder_glove(x):
    
    print("Loading encoder model...")
    enc = encoder.Autoencoder_glove()
    print('encoder Loaded Successfully')
    
    print("training Encoder")
    trained_enc = encoder_training.train(enc, 25, x)
    print(' Encoder Trained Successfully')
    return trained_enc

def load_model1(Path):
    
    model1 = model.VidSmodel()
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model2(Path):
    
    model1 = model.VidSmodel2()
    
    model1 = torch.load(Path)
    model1.eval()
    return model1

def load_model2_glove(Path):
    
    model1 = model.VidSmodel2_glove()
    
    model1 = torch.load(Path)
    model1.eval()
    return model1

def load_model2_denseNet(Path):
    
    model1 = model.VidSmodel2_DenseNet()
    
    model1 = torch.load(Path)
    model1.eval()
    return model1

def load_model2_denseNet_glove(Path):
    
    model1 = model.VidSmodel2_glove_denseNet()
    
    model1 = torch.load(Path)
    model1.eval()
    return model1

def load_model3(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    
    model1 = model.VidSmodel3(enc)
    
    model1 = torch.load(Path)
    model1.eval()
    return model1

def load_model4(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    
    model1 = model.VidSmodel4(enc)
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model4_glove(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1_glove.pt")
    enc.eval()
    
    model1 = model.VidSmodel4(enc)
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model4_denseNet(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    
    model1 = model.VidSmodel4_DenseNet(enc)
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model4_denseNet_glove(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1_glove.pt")
    enc.eval()
    
    model1 = model.VidSmodel4_DenseNet(enc)
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model5(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    
    model1 = model.VidSmodel5(enc)
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model5_glove(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1_glove.pt")
    enc.eval()
    
    model1 = model.VidSmodel5(enc)
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model5_denseNet(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    
    model1 = model.VidSmodel5_DenseNet(enc)
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model5_denseNet_glove(Path):
    
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1_glove.pt")
    enc.eval()
    
    model1 = model.VidSmodel5_DenseNet(enc)
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model6(Path):
    
    model1 = model.VidSmodel6()
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def load_model7(Path):
    
    model1 = model.VidSmodel7()
    
    model1 = torch.load(Path)
    #model1.eval()
    return model1

def train_model2_glove(glove, path, logsPath):
    
    print('Loading Model2...')
    model1 = model.VidSmodel2_glove()
    print('Model Loaded Successfully')
    
    print('Training Model...Cosine Similarity with 0.005l2 and step decay per 20 epochs')
    trained_model = train.trainNet(model1, 10, 30, 0.001, glove, path, logsPath, 20, 0.005, 'train3.txt')
    print(' Model Trained Successfully')
    return trained_model

def train_model4_glove(glove, path, logsPath):
    
    print('Loading Model4...')
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1_glove.pt")
    enc.eval()
    model1 = model.VidSmodel4(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...Cosine Similarity with 0.01l2 and step decay per 15 epochs')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 50, 0.001, glove, path, logsPath, 15, 0.001, 'train2.txt')
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model5_glove(glove, path, logsPath):
    
    print('Loading Model5....cosine similarity with 0.001l2')
    enc = encoder.Autoencoder_glove()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1_glove.pt")
    enc.eval()
    model1 = model.VidSmodel5(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...Cosine Similarity with 0.001l2')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 35, 0.0001, glove, path, logsPath, 50, 0.001, 'train1.txt')
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model5_denseNet(w2vmodel, path, logsPath):
    
    print('Loading Model5....DenseNet cosine similarity with 0.001l2')
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    model1 = model.VidSmodel5_DenseNet(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...DenseNet and word2vec Cosine Similarity with 0.001l2')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 35, 0.0001, w2vmodel, path, logsPath, 50, 0.001, 'train1.txt')
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model4_densenet(w2vmodel, path, logsPath):
    
    print('Loading Model4...DenseNet')
    enc = encoder.Autoencoder()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1.pt")
    enc.eval()
    model1 = model.VidSmodel4_DenseNet(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...DenseNet and word2vec Cosine Similarity with 0.01l2 and step decay per 15 epochs')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 50, 0.0001, w2vmodel, path, logsPath, 15, 0.01, 'train2.txt')
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model2_densenet(w2vmodel, path, logsPath):
    
    print('Loading Model2...DenseNet')
    model1 = model.VidSmodel2_DenseNet()
    print('Model Loaded Successfully')
    
    print('Training Model...DenseNet and word2vec Cosine Similarity with 0.001l2')
    trained_model, acc_list, loss_list  = train.trainNet(model1, 10, 30, 0.0001, w2vmodel, path, logsPath, 30, 0.001, 'train3.txt')
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model5_denseNet_glove(glove, path, logsPath):
    
    print('Loading Model5....DenseNet cosine similarity with 0.001l2')
    enc = encoder.Autoencoder_glove()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1_glove.pt")
    enc.eval()
    model1 = model.VidSmodel5_DenseNet(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...DenseNet and glove Cosine Similarity with 0.001l2')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 35, 0.0001, glove, path, logsPath, 50, 0.001, 'train1.txt')
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model4_densenet_glove(glove, path, logsPath):
    
    print('Loading Model4...DenseNet')
    enc = encoder.Autoencoder_glove()
    enc = torch.load("/content/drive/MyDrive/New_Code/New model/encoder_model1_glove.pt")
    enc.eval()
    model1 = model.VidSmodel4_DenseNet(enc)
    print('Model Loaded Successfully')
    
    print('Training Model...DenseNet and glove Cosine Similarity with 0.01l2 and step decay per 15 epochs')
    trained_model, acc_list, loss_list = train.trainNet(model1, 10, 50, 0.0001, glove, path, logsPath, 15, 0.01, 'train2.txt')
    print(' Model Trained Successfully')
    return trained_model, acc_list, loss_list

def train_model2_densenet_glove(glove, path, logsPath):
    
    print('Loading Model2...DenseNet')
    model1 = model.VidSmodel2_glove_denseNet()
    print('Model Loaded Successfully')
    
    print('Training Model...DenseNet and glove Cosine Similarity with 0.005l2')
    trained_model = train.trainNet(model1, 10, 30, 0.0001, glove, path, logsPath, 30, 0.005, 'train3.txt')
    print(' Model Trained Successfully')
    return trained_model
    
    
