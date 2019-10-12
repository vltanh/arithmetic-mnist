import time
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.nn import functional as F

from dataset import MNISTDataset
from model import Classifier

def get_dataset():
    datasets = dict()
    trainval_dataset = MNISTDataset(path='data/train.csv', is_train=True)
    train_ratio = 0.8
    train_size = int(train_ratio * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    datasets['train'], datasets['val'] = data.random_split(trainval_dataset, 
                                                            [train_size, val_size])

    datasets['test'] = MNISTDataset(path='data/test.csv', is_train=False)

    dataloaders = dict()
    dataloaders = { split: data.DataLoader(dataset, batch_size=4, shuffle=True) 
                    for split, dataset in datasets.items() }

    return dataloaders

def get_network():
    return Classifier()

def get_loss():
    return nn.CrossEntropyLoss()

def post_processing(outs):
    _, preds = torch.max(F.softmax(outs, dim=1), dim=1)
    return preds

def accuracy(lbls, preds):
    return torch.sum(lbls == preds).item()

def train_phase(net, dataloader, optimizer, criterion, log_step, device):
    # Record loss and metrics during training
    running_loss = 0.0
    running_acc = 0.0

    # Start training
    net.train()
    for i, (inps, lbls) in enumerate(dataloader):
        # Load inputs and labels
        inps = inps.to(device)
        lbls = lbls.to(device)

        # Clear out gradients from previous iteration
        optimizer.zero_grad()
        # Get network outputs
        outs = net(inps)
        # Calculate the loss
        loss = criterion(outs, lbls)
        # Calculate the gradients
        loss.backward()
        # Performing backpropagation
        optimizer.step()
        
        # Update loss
        running_loss += loss.item()

        # Post processing the outputs
        preds = post_processing(outs.detach())
        
        # Update metrics
        running_acc += accuracy(lbls, preds) / len(lbls)

        # Log in interval
        if (i + 1) % log_step == 0:
            print('Iter {:>5d}, loss: {:.5f}, acc: {:.5f}'.format(
                i + 1, running_loss / log_step, running_acc / log_step))
            running_loss = 0.0
            running_acc = 0.0

def val_phase(net, dataloader, criterion, logger, device):
    # No calculating gradients
    with torch.no_grad():
        start = time.time()
        
        # Record loss and metrics
        _loss = 0.0
        _acc = 0.0
        
        # Start validating
        net.eval()
        for i, (inps, lbls) in enumerate(dataloader):
            # Load inputs and labels
            inps = inps.to(device)
            lbls = lbls.to(device)
            
            # Get network outputs
            outs = net(inps)
            
            # Calculate the loss
            loss = criterion(outs, lbls)
            
            # Update loss
            _loss += loss.item()

            # Post processing outputs
            preds = post_processing(outs)

            # Update metrics
            _acc += accuracy(lbls, preds)

    # Calculate evaluation result
    datasize = len(dataloader.dataset)
    avg_pred_time = (time.time() - start) / datasize
    avg_loss = _loss / datasize
    avg_acc = _acc / datasize
    
    # Print results
    print('-' * 20)
    print('Average prediction time: {} (s)'.format(avg_pred_time))
    print('Loss:', avg_loss)
    print('Acc:', avg_acc)
    
    # Log loss and metrics
    logger['loss'].append(avg_loss)
    logger['acc'].append(avg_acc)

def main():
    # Specify device
    dev_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(dev_id)

    # Load datasets
    dataloaders = get_dataset()

    # Define network
    net = get_network().to(dev)
    print(net)

    # Define loss
    criterion = get_loss()

    # Define optim
    optimizer = optim.SGD(net.parameters(), 
                          lr=0.0001, 
                          momentum=0.99)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     patience=5, 
                                                     verbose=True)

    # Prepare logger
    logger = dict()
    for phase in ['train', 'val']:
        logger[phase] = dict()
        logger[phase]['loss'] = []
        logger[phase]['acc'] = []
    logger['best'] = dict()
    logger['best']['val_loss'] = 100000.0
    logger['best']['val_acc'] = 0.0
    
    # -----------------------------------------------------------------
    
    # Start training loop
    nepochs = 1
    for epoch in range(nepochs):
        print('=' * 30)
        print('Epoch {:>3d}'.format(epoch))
        
        # --------------------------------------------------------------
        
        # Training phase
        print('-' * 20)
        print('Start [training].')
        start = time.time()
        
        train_phase(net=net, 
                    dataloader=dataloaders['train'], 
                    optimizer=optimizer,
                    criterion=criterion,
                    log_step=100,
                    device=dev)
        
        print('Training takes %f (s)' % (time.time() - start))

        # --------------------------------------------------------------
        
        # Validation phase
        print('-' * 20)
        print('Start [validating].')
        start = time.time()
        
        # Validate on train dataset
        val_phase(net=net,
                 dataloader=dataloaders['train'],
                 criterion=criterion,
                 logger=logger['train'],
                 device=dev)
        
        # Validate on val dataset
        val_phase(net=net,
                 dataloader=dataloaders['val'],
                 criterion=criterion,
                 logger=logger['val'],
                 device=dev)
        
        print('Validating takes %f (s)' % (time.time() - start))
        
        # Getting val_loss and val_acc for this epoch
        val_loss = logger['val']['loss'][-1]
        val_acc = logger['val']['acc'][-1]
        
        # --------------------------------------------------------------s
        
        # Learning rate scheduling based on last val_loss
        scheduler.step(val_loss)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Exiting...')