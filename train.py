import time
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.nn import functional as F

from dataset import MNISTDataset
from model import Classifier, FeatureExtractor

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
    dataloaders = { split: data.DataLoader(dataset, 
                                           batch_size=2, 
                                           shuffle=True,
                                           drop_last=True) 
                    for split, dataset in datasets.items() }

    return dataloaders

def get_network():
    return FeatureExtractor(), Classifier()

def get_loss():
    return nn.CrossEntropyLoss()

def post_processing(outs):
    _, preds = torch.max(F.softmax(outs, dim=1), dim=1)
    return preds

def accuracy(lbls, preds):
    return torch.sum(lbls == preds).item()

def train_phase(net, dataloader, optimizer, criterion, log_step, device):
    fe, cl = net
    fe_opt, cl_opt = optimizer

    # Record loss and metrics during training
    running_loss = 0.0
    running_acc = 0.0

    # Start training
    fe.train()
    cl.train()
    for i, (inps, lbls) in enumerate(dataloader):
        # Load inputs and labels
        inps = inps.to(device)
        lbls = lbls.to(device)

        with torch.no_grad():
            inp_1 = inps[0]
            inp_2 = inps[1]
            lbl_1 = lbls[0]
            lbl_2 = lbls[1]

            if lbl_1 < lbl_2:
                inp_2, inp_1 = inp_1, inp_2
                lbl_1, lbl_2 = lbl_2, lbl_1

            inp_1 = inp_1[None, :, :, :]
            inp_2 = inp_2[None, :, :, :]
            
            lbls = torch.Tensor([abs(lbl_1 - lbl_2)]).long().to(device)
            lbl_1 = torch.Tensor([lbl_1]).long().to(device)
            lbl_2 = torch.Tensor([lbl_2]).long().to(device)

        # Clear out gradients from previous iteration
        fe_opt.zero_grad()
        cl_opt.zero_grad()
        # Get network outputs
        out_1 = fe(inp_1)
        out_2 = fe(inp_2)
        # outs = cl(torch.cat([out_1, out_2], dim=1))
        outs = cl(out_1 - out_2)
        # Calculate the loss
        loss = criterion(outs, lbls)
        # loss += 0.01 * (criterion(out_1, lbl_1) + criterion(out_2, lbl_2))
        # Calculate the gradients
        loss.backward()
        # Performing backpropagation
        fe_opt.step()
        cl_opt.step()
        
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
    fe, cl = net

    # No calculating gradients
    with torch.no_grad():
        start = time.time()
        
        # Record loss and metrics
        _loss = 0.0
        _acc = 0.0
        
        # Start validating
        fe.eval()
        cl.eval()
        for i, (inps, lbls) in enumerate(dataloader):
            # Load inputs and labels
            inps = inps.to(device)
            lbls = lbls.to(device)

            with torch.no_grad():
                inp_1 = inps[0]
                inp_2 = inps[1]
                lbl_1 = lbls[0]
                lbl_2 = lbls[1]

                if lbl_1 < lbl_2:
                    inp_2, inp_1 = inp_1, inp_2
                    lbl_1, lbl_2 = lbl_2, lbl_1

                inp_1 = inp_1[None, :, :, :]
                inp_2 = inp_2[None, :, :, :]

                lbls = torch.Tensor([abs(lbl_1 - lbl_2)]).long().to(device)
                lbl_1 = torch.Tensor([lbl_1]).long().to(device)
                lbl_2 = torch.Tensor([lbl_2]).long().to(device)

            # Get network outputs
            out_1 = fe(inp_1)
            out_2 = fe(inp_2)
            # outs = cl(torch.cat([out_1, out_2], dim=1))
            outs = cl(out_1 - out_2)
            
            # Calculate the loss
            loss = criterion(outs, lbls)
            # loss += 0.01 * (criterion(out_1, lbl_1) + criterion(out_2, lbl_2))
            
            # Update loss
            _loss += loss.item()

            # Post processing outputs
            preds = post_processing(outs)

            # Update metrics
            _acc += accuracy(lbls, preds)

    # Calculate evaluation result
    datasize = len(dataloader.dataset) // 2
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
    fe, cl = get_network()
    fe = fe.to(dev)
    cl = cl.to(dev)

    # Define loss
    criterion = get_loss()

    # Define optim
    fe_opt = optim.SGD(fe.parameters(), 
                       lr=0.0001, 
                       momentum=0.99)
    cl_opt = optim.SGD(cl.parameters(), 
                       lr=0.0001, 
                       momentum=0.99)
    
    # Define learning rate scheduler
    fe_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fe_opt, 
                                                        patience=3, 
                                                        verbose=True)
    cl_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cl_opt, 
                                                        patience=3, 
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
    nepochs = 50
    for epoch in range(nepochs):
        print('=' * 30)
        print('Epoch {:>3d}'.format(epoch))
        
        # --------------------------------------------------------------
        
        # Training phase
        print('-' * 20)
        print('Start [training].')
        start = time.time()
        
        train_phase(net=[fe, cl], 
                    dataloader=dataloaders['train'], 
                    optimizer=[fe_opt, cl_opt],
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
        val_phase(net=[fe, cl],
                 dataloader=dataloaders['train'],
                 criterion=criterion,
                 logger=logger['train'],
                 device=dev)
        
        # Validate on val dataset
        val_phase(net=[fe, cl],
                 dataloader=dataloaders['val'],
                 criterion=criterion,
                 logger=logger['val'],
                 device=dev)
        
        print('Validating takes %f (s)' % (time.time() - start))
        
        # Getting val_loss and val_acc for this epoch
        val_loss = logger['val']['loss'][-1]
        val_acc = logger['val']['acc'][-1]
        
        # --------------------------------------------------------------
        
        # Learning rate scheduling based on last val_loss
        fe_scheduler.step(val_loss)
        cl_scheduler.step(val_loss)

        # --------------------------------------------------------------

        save_info = {
            'fe_state_dict': fe.state_dict(),
            'fe_opt_state_dict': fe_opt.state_dict(),
            'cl_state_dict': cl.state_dict(),
            'cl_opt_state_dict': cl_opt.state_dict()
        }

        if val_loss < logger['best']['val_loss']:
            torch.save(save_info, 'weights/best_loss.pth')
            logger['best']['val_loss'] = val_loss

        if val_acc > logger['best']['val_acc']:
            torch.save(save_info, 'weights/best_acc.pth')
            logger['best']['val_acc'] = val_acc

        # --------------------------------------------------------------
        
        # Plot training graph
        for metric in ['loss', 'acc']:
            plt.figure()
            plt.plot(range(len(logger['train'][metric])), logger['train'][metric])
            plt.plot(range(len(logger['val'][metric])), logger['val'][metric])
            plt.legend(['train', 'val'])
            plt.savefig(os.path.join('logs', f'{metric}'))
            plt.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Exiting...')