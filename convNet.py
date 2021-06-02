import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.nn as nn
import torch.nn.functional as F

H = 28
W = 28
BATCH_SIZE = 4
INPUT_SIZE = H * W

def plotBatch(dim, batch):
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, dim[0] * dim[1] +1):
        img = batch[i - 1].squeeze(0)
        fig.add_subplot(dim[0], dim[1], i)
        plt.imshow(img)
    plt.show()

# Load dataset

class MNISTDataSet(Dataset):
    def __init__(self, file_path):
        d = pd.read_csv(file_path)
        self.labels = torch.tensor(d['label'], dtype=torch.float)
        self.data = torch.tensor(d.drop('label', axis=1).values, dtype=torch.float).view(-1, 1, H, W)
        return
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data' : self.data[index], 'label' : self.labels[index]}

# Preprocess data

# Create Network

class ConvNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        return

    def forward(self):
        return

# Train

# Evaluate

if __name__ == "__main__":
    mnist = MNISTDataSet('train.csv')
    batch = DataLoader(mnist, 16)

    for i, sample in enumerate(batch):
        if i == 0:
            plotBatch((4, 4), sample['data'])
            break