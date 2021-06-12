import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, dataloader, random_split
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, file_path, test=False):
        d = pd.read_csv(file_path)
        if not test:
            self.labels = torch.tensor(d['label'], dtype=torch.long)
            self.data = torch.tensor(d.drop('label', axis=1).values, dtype=torch.float).view(-1, 1, H, W)
        else:
            self.labels = torch.empty(len(d), dtype=torch.float)
            self.data = torch.tensor(d.values, dtype=torch.float).view(-1, 1, H, W)
        return
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data' : self.data[index], 'label' : self.labels[index]}

# Preprocess data

# Create Network

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 10)
        return

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Train

train_set, val_set = random_split(MNISTDataSet('train.csv'), [35000, 7000])
batch = DataLoader(train_set, 16)
net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
avg_loss = 0
for i, sample in enumerate(batch):
    optimizer.zero_grad()
    pred = net(sample['data'])
    loss = criterion(pred, sample['label'])
    avg_loss += loss
    if (i + 1) % 100 == 0:
        avg_loss /= 100
        print(f"Batch {i} Avg Loss {avg_loss}")
        avg_loss = 0
    loss.backward()
    optimizer.step()

# Evaluate

with torch.no_grad():
    outputs = net(val_set[:]['data'])
    _, predicted = torch.max(outputs, 1)
    correct = predicted == val_set[:]['label']
    accuracy = sum(correct) / len(correct)
    print(accuracy)

# Create Kaggle submission

mnist_test = MNISTDataSet('test.csv', test=True)
with torch.no_grad():
    outputs = net(mnist_test[:]['data'])
    _, predicted = torch.max(outputs, 1)

submission = pd.DataFrame(predicted, columns=["Label"])
submission.index += 1
submission.to_csv("submission.csv", sep=",", index_label="ImageId")