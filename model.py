import torch
from torch import nn
from torch.nn import functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=7*7*128, out_features=128)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # self.feature_extractor = FeatureExtractor()
        self.fc1 = nn.Linear(in_features=128, out_features=10)
    
    def forward(self, x):
        # x = self.feature_extractor(x)
        x = self.fc1(x)
        return x