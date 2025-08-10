"""
   kuzu.py
   ZZEN9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERTED CODE HERE # Anirban
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 784)
        x = self.fc(x)  # logits
        return F.log_softmax(x, dim=1)
        # return 0 # CHANGED CODE HERE # Anirban

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    # def __init__(self):
    #     super(NetFull, self).__init__()
        # INSERTED CODE HERE # Anirban
    def __init__(self, hidden=160):  # Tried values like 30, 60, 80, 90, 100, 110, 120, 140, 160...
        super(NetFull, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
    #     return 0 # CHANGED CODE HERE # Anirban
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        # super(NetConv, self).__init__()
        # INSERTED CODE HERE # Anirban
        super(NetConv, self).__init__()
        # First conv layer: 1 input channel, 32 output channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 28x28 → 28x28
        # Second conv layer: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 28x28 → 28x28
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 → 14x14
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        # return 0 # CHANGED CODE HERE # Anirban
        x = F.relu(self.conv1(x))  # (B,32,28,28)
        x = F.relu(self.conv2(x))  # (B,64,28,28)
        x = self.pool(x)  # (B,64,14,14)
        x = x.view(x.size(0), -1)  # flatten (B, 64*14*14)
        x = F.relu(self.fc1(x))  # (B,128)
        x = self.out(x)  # (B,10)
        return F.log_softmax(x, dim=1)
