import time
import matplotlib.pyplot as plt
import os

import torch
from torch.utils import data
from torch import nn
from torch.nn import functional as F

from dataset import MNISTDataset
from model import DifferenceClassifier, FeatureExtractor

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
    return FeatureExtractor(), DifferenceClassifier()

def get_loss():
    return nn.MSELoss()

def post_processing(outs):
    return outs

def pre_get_diff(inps, lbls, device):
    with torch.no_grad():
        inp_1 = inps[0][None, :, :, :]
        inp_2 = inps[1][None, :, :, :]
        lbl_1 = lbls[0]
        lbl_2 = lbls[1]
        lbls = torch.Tensor([lbl_1 - lbl_2]).to(device)[None, :]
    return inp_1, inp_2, lbls

def val_phase(net, dataloader, criterion, output_path, device):
    os.system(f'mkdir -p {output_path}')

    fe, cl = net

    # No calculating gradients
    with torch.no_grad():
        start = time.time()
        
        # Record loss and metrics
        _loss = 0.0
        
        # Start validating
        fe.eval()
        cl.eval()
        for i, (inps, lbls) in enumerate(dataloader):
            # Load inputs and labels
            inps = inps.to(device)
            lbls = lbls.to(device)

            inp_1, inp_2, lbls = pre_get_diff(inps, lbls, device)

            # Get network outputs
            out_1 = fe(inp_1)
            out_2 = fe(inp_2)
            # outs = cl(torch.cat([out_1, out_2], dim=1))
            outs = cl(out_1 - out_2)
            
            # Calculate the loss
            loss = criterion(outs, lbls)

            # Update loss
            _loss += loss.item()

            # Post processing outputs
            preds = post_processing(outs)

            plt.subplot(1, 2, 1)
            plt.imshow(inp_1.detach().cpu().squeeze())
            plt.subplot(1, 2, 2)
            plt.imshow(inp_2.detach().cpu().squeeze())
            plt.suptitle(f'{preds.item()}')
            plt.savefig(f'{output_path}/{i:04d}')
            plt.close()

    # Calculate evaluation result
    datasize = len(dataloader.dataset) // 2
    avg_pred_time = (time.time() - start) / datasize
    avg_loss = _loss / datasize
    
    # Print results
    print('-' * 20)
    print('Average prediction time: {} (s)'.format(avg_pred_time))
    print('Loss:', avg_loss)

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

    pretrained = torch.load('weights/mse_best_loss.pth', map_location=dev_id)
    fe.load_state_dict(pretrained['fe_state_dict'])
    cl.load_state_dict(pretrained['cl_state_dict'])

    # Define loss
    criterion = get_loss()

    # Validate on val dataset
    val_phase(net=[fe, cl],
             dataloader=dataloaders['val'],
             criterion=criterion,
             output_path='visualization',
             device=dev)

if __name__ == "__main__":
    main()