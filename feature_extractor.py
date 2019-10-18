import time
import matplotlib.pyplot as plt
import pickle
import numpy as np

import torch
from torch.utils import data
from torch import nn
from torch.nn import functional as F

from dataset import MNISTDataset
from model import FeatureExtractor

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
                                           batch_size=4, 
                                           shuffle=True) 
                    for split, dataset in datasets.items() }

    return dataloaders

def get_network():
    return FeatureExtractor()

def feature_extract(net, dataloader, output_path, device):
    dataset = dict()
    dataset['data'] = []
    dataset['features'] = []
    dataset['labels'] = []

    # No calculating gradients
    with torch.no_grad():
        start = time.time()
        
        # Start validating
        net.eval()
        for i, (inps, lbls) in enumerate(dataloader):
            # Load inputs and labels
            inps = inps.to(device)
            lbls = lbls.to(device)

            # Get network outputs
            outs = net(inps)

            dataset['data'].extend(inps.detach().cpu().squeeze().view(4, -1).numpy())
            dataset['features'].extend(outs.detach().cpu().numpy())
            dataset['labels'].extend(lbls.detach().cpu().numpy())
    
    dataset['data'] = np.array(dataset['data'])
    dataset['features'] = np.array(dataset['features'])
    dataset['labels'] = np.array(dataset['labels'])
    pickle.dump(dataset, open(output_path, 'wb'))


def main():
    # Specify device
    dev_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(dev_id)

    # Load datasets
    dataloaders = get_dataset()

    # Define network
    fe = get_network()
    fe = fe.to(dev)

    pretrained = torch.load('weights/cl_best_loss.pth', map_location=dev_id)
    fe.load_state_dict(pretrained['fe_state_dict'])

    # Validate on val dataset
    for x in ['train', 'val', 'test']:
        feature_extract(net=fe,
                        dataloader=dataloaders[f'{x}'],
                        output_path=f'features/{x}.dat',
                        device=dev)

if __name__ == "__main__":
    main()