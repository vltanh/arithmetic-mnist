import time
import matplotlib.pyplot as plt

import torch
from torch.utils import data
from torch import nn
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

def val_phase(net, dataloader, criterion, output_path, device):
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
            pred_1 = post_processing(cl(out_1))
            pred_2 = post_processing(cl(out_2))
            preds = post_processing(outs)

            # Update metrics
            _acc += accuracy(lbls, preds)

            plt.subplot(1, 2, 1)
            plt.imshow(inp_1.detach().cpu().squeeze())
            plt.title(f'{pred_1.item()}')
            plt.subplot(1, 2, 2)
            plt.imshow(inp_2.detach().cpu().squeeze())
            plt.title(f'{pred_2.item()}')
            plt.suptitle(f'{preds.item()}')
            plt.savefig(f'{output_path}/{i:04d}')
            plt.close()

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

    pretrained = torch.load('weights/best_loss__.pth', map_location=dev_id)
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