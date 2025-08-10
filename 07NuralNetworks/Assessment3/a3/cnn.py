#!/usr/bin/env python3
"""
   cnn.py

   UNSW ZZEN9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
a3main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing.
    Images are 80x80 RGB.
    """
    # if mode == 'train':
    #     return transforms.ToTensor()
    # elif mode == 'test':
    #     return transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])  # scale to [-1,1]

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.RandomResizedCrop(80, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.CenterCrop(80),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Fallback (shouldn't be used by a3main)
        return transforms.Compose([transforms.Resize((80, 80)),
                                   transforms.ToTensor(),
                                   normalize])

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, p, p_drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_drop),
            nn.MaxPool2d(2)  # halves H and W
        )

    def forward(self, x):
        return self.block(x)


class Network(nn.Module):

    # def __init__(self):
    #     super().__init__()
    def __init__(self):
        super().__init__()
        # Input: (B,3,80,80)
        # *********************************
        # Model #1 400KB 40% accuracy #
        # self.layer1 = ConvBlock(3,   32, p_drop=0.10)  # -> (B,32,40,40)
        # self.layer2 = ConvBlock(32,  64, p_drop=0.15)  # -> (B,64,20,20)
        # self.layer3 = ConvBlock(64, 128, p_drop=0.20)  # -> (B,128,10,10)
        #
        # # Global Average Pooling to remove dependence on spatial size
        # self.gap = nn.AdaptiveAvgPool2d(1)            # -> (B,128,1,1)
        # self.classifier = nn.Linear(128, 8)           # 8 cat breeds
        # *********************************

        # *********************************
        # Model #2: 6 Conv layers #
        self.conv1 = ConvBNReLU(3, 32, k=5, p=2, p_drop=0.05)  # -> (B,32,80,80)
        self.pool1 = nn.MaxPool2d(2)  # -> (B,32,40,40)

        self.conv2 = ConvBNReLU(32, 64, k=5, p=2, p_drop=0.05)  # -> (B,64,40,40)

        # 3 & 4: 5x5
        self.conv3 = ConvBNReLU(64, 128, k=5, p=2, p_drop=0.05)  # -> (B,128,40,40)
        self.pool3 = nn.MaxPool2d(2)  # -> (B,128,20,20)

        self.conv4 = ConvBNReLU(128, 128, k=3, p=1, p_drop=0.10)  # -> (B,128,20,20)

        # 5 & 6: 3x3
        self.conv5 = ConvBNReLU(128, 256, k=3, p=1, p_drop=0.10)  # -> (B,256,20,20)
        self.pool5 = nn.MaxPool2d(2)  # -> (B,256,10,10)

        self.conv6 = ConvBNReLU(256, 256, k=3, p=1, p_drop=0.10)  # -> (B,256,10,10)

        # Global Average Pooling to one vector per channel
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B,256,1,1)

        # Single fully connected classifier layer
        self.classifier = nn.Linear(256, 8)
    # *********************************

    # def forward(self, input):
    #     pass
    # *********************************
    # Model #1 400KB 40% accuracy #
    # def forward(self, x):
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.gap(x)               # (B,128,1,1)
    #     x = torch.flatten(x, 1)       # (B,128)
    #     logits = self.classifier(x)   # (B,8)
    #     return logits
    # *********************************
    # Model # 2 6 convolution layers
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)

        x = self.conv5(x)
        x = self.pool5(x)

        x = self.conv6(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)  # (B, 256)
        logits = self.classifier(x)  # (B, 8)
        return logits
    # *********************************


# Instantiate the network
net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
# optimizer = None
# Adam with modest weight decay for regularization
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

# loss_func = None
# CrossEntropyLoss expects raw logits (no softmax in forward)
loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    # return
    # Kaiming initialization for ReLU conv/linear; zeros for biases; BN to ones/zeros
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# scheduler = None
# StepLR: decay LR to encourage convergence after initial progress
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 128        # 200
epochs = 25             # 10
