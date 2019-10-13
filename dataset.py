import pandas as pd

import torch
from torch.utils import data

class MNISTDataset(data.Dataset):
    def __init__(self, path, is_train):
        self.is_train = is_train

        # Load dataset from csv file
        data = pd.read_csv(path)

        if self.is_train:
            # Get label column
            self.labels = data['label'].values
            # Get data from the rest (by dropping the first column - label)
            self.inputs = data.drop(labels=['label'], axis=1).values
        else:
            self.inputs = data.values

        del data

    def __getitem__(self, i):
        inp = self.preprocess_input(self.inputs[i])
        if self.is_train:
            lbl = self.preprocess_label(self.labels[i])
            return inp, lbl
        else:
            return inp

    def __len__(self):
        return len(self.inputs)

    def preprocess_input(self, inp):
        return torch.from_numpy(inp / 255.0).view((28, 28)).unsqueeze(0).float()

    def preprocess_label(self, lbl):
        return lbl
