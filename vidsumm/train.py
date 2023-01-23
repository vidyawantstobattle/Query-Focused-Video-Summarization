import torch
import time
import torch.optim as optim
from torch.autograd import Variable
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.tensorboard import SummaryWriter
from . import dataset
from . import transformer

def trainNet(model, batch_size, n_epochs, learning_rate, text_model, path, logsPath, step_epoch, decay, logFile):
    
    # For GPU
    net = model.cuda()
    net.train()  
    
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    data_loader = dataset.dataset("/content/drive/MyDrive/New_Code/New model/training_data.csv","/content/drive/MyDrive/New_Code/New model/dataset/frames", text_model, transformer.train_transform)
    n_batches = len(data_loader)
    
    params = {'batch_size': 10,
          'shuffle': True}
    
    train_data = torch.utils.data.DataLoader(data_loader, **params)
    
    accuracy_list = []
    loss_list = []

    loss = torch.nn.CrossEntropyLoss()
    
    # for param in net.enc.parameters():
    #     param.requires_grad = False

    for param in net.model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = decay)
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_epoch, gamma = 0.1, verbose=True)
    training_start_time = time.time()
    
    log_file = open(logFile,'w') 

    writer = SummaryWriter(logsPath)
    
    for epoch in range(n_epochs):
        
        count_relevance = 0
        running_loss = 0.0
        start_time = time.time()  
        total_train_loss = 0
        
        i_batch = 0
        
        for sample_batched in train_data:
            
    
            #Get inputs
            inputs, query, labels = sample_batched['image'], sample_batched['query'], sample_batched['score_annotations']  
            
            
            labels_relevance = labels[:,0]
            
            
            inputs = inputs.cuda()
            query = query.cuda()
            labels_relevance = labels_relevance.cuda()
            
            #Wrap them in a Variable object
            #inputs, query, labels_relevance = Variable(inputs), Variable(query), Variable(labels_relevance)
            
            optimizer.zero_grad()
            
            outputs = net(inputs, query)
            loss_size_1 = loss(outputs, labels_relevance.long())
            loss_size = loss_size_1
            loss_size.backward()
            optimizer.step()  
   
            running_loss += loss_size.item()
    
            total_train_loss += loss_size.item()  
           
            #Compute accuracy
            max_values_relevance, arg_maxs_relevance = torch.max(outputs, dim = 1)
            num_correct_relevance = torch.sum(labels_relevance.long() == arg_maxs_relevance.long())
            count_relevance = count_relevance + num_correct_relevance.item()
        
        
            print("Epoch {}, {:d}% \t train_loss_{}_batch: {:.4f} \t took: {:.4f}s".format(
                        epoch+1, int(100 * (i_batch+1) / len(train_data)), i_batch+1, running_loss, time.time() - start_time))

            log_file.write("Epoch {}, {:d}% \t train_loss_{}_batch: {:.4f} \t took: {:.4f}s \n".format(
                        epoch+1, int(100 * (i_batch+1) / len(train_data)), i_batch+1, running_loss, time.time() - start_time))
            log_file.flush()

            running_loss = 0.0
            i_batch = i_batch+1
            start_time = time.time()

        acc_relevance = (float(count_relevance)/(len(data_loader)))
        print("Training accuracy_relevance = {:.4f} for epoch {}".format(acc_relevance, epoch +1))
        accuracy_list.append(acc_relevance)
        log_file.write("Training accuracy_relevance = {:.4f} for epoch {} \n".format(acc_relevance, epoch +1))
        
        total_train_loss = (float(total_train_loss)/(len(train_data)))
        print("Training loss = {:.4f} for epoch {}".format(total_train_loss, epoch +1))
        loss_list.append(total_train_loss)
        log_file.write("Training loss = {:.4f} for epoch {} \n".format(total_train_loss, epoch +1))
        
        log_file.flush()

        lr_scheduler.step()

        writer.add_scalar('training loss', total_train_loss, epoch+1)
        writer.add_scalar('training accuracy', acc_relevance, epoch+1)
        writer.flush()

    print("Training finished, took {:.4f}s".format(time.time() - training_start_time))
    log_file.write("Training finished, took {:.4f}s \n".format(time.time() - training_start_time))
    
    log_file.close() 
    writer.close()

    # state = {
    # 'epoch': n_epochs,
    # 'state_dict': net.state_dict(),
    # 'optimizer': optimizer.state_dict(),
    # 'scheduler': lr_scheduler.state_dict()
    # }
    # torch.save(state, path)
    
    return net, accuracy_list, loss_list

def valNet(model, batch_size, n_epochs, learning_rate, w2vmodel):
    
    # For GPU
    net = model.cuda()
    net.train()  
    
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get data
    data_loader = dataset.dataset("/content/drive/MyDrive/New_Code/New model/val_data.csv","/content/drive/MyDrive/New_Code/New model/dataset/frames", w2vmodel, transformer.valid_transform)
    n_batches = len(data_loader)
    
    params = {'batch_size': 10,
          'shuffle': True}
    
    train_data = torch.utils.data.DataLoader(data_loader, **params)

    loss = torch.nn.CrossEntropyLoss()

    #optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    
    training_start_time = time.time()
    
    log_file = open('validation_info.txt','w') 
    
    for epoch in range(n_epochs):
        
        count_relevance = 0
        running_loss = 0.0
        start_time = time.time()  
        total_train_loss = 0
        
        i_batch = 0
        for sample_batched in train_data:
            
    
            #Get inputs
            inputs, query, labels = sample_batched['image'], sample_batched['query'], sample_batched['score_annotations']  
            
            
            labels_relevance = labels[:,0]
            

            inputs = inputs.cuda()
            query = query.cuda()
            labels_relevance = labels_relevance.cuda()
            
            #Wrap them in a Variable object
            inputs, query, labels_relevance = Variable(inputs), Variable(query), Variable(labels_relevance)
            
            
            outputs = net(inputs, query)

            loss_size_1 = loss(outputs, labels_relevance.long())
            loss_size = loss_size_1
   
            running_loss += loss_size.item()
    
            total_train_loss += loss_size.item()  
           
            #Compute accuracy
            max_values_relevance, arg_maxs_relevance = torch.max(outputs, dim = 1)
            num_correct_relevance = torch.sum(labels_relevance.long() == arg_maxs_relevance.long())
            count_relevance = count_relevance + num_correct_relevance.item()
        
        
            print("Epoch {}, {:d}% \t val_loss_{}_batch: {:.4f} \t took: {:.4f}s".format(
                        epoch+1, int(100 * (i_batch+1) / len(train_data)), i_batch+1, running_loss, time.time() - start_time))

            log_file.write("Epoch {}, {:d}% \t val_loss_{}_batch: {:.4f} \t took: {:.4f}s \n".format(
                        epoch+1, int(100 * (i_batch+1) / len(train_data)), i_batch+1, running_loss, time.time() - start_time))
            log_file.flush()

            running_loss = 0.0
            i_batch = i_batch+1
            start_time = time.time()

        acc_relevance = (float(count_relevance)/(len(data_loader)))
        print("Validation accuracy_relevance = {:.4f} for epoch {}".format(acc_relevance, epoch +1))


    print("Validation finished, took {:.4f}s".format(time.time() - training_start_time))
    
    log_file.close() 
    return net