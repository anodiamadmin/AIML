#!/usr/bin/env python3
"""
   student.py

   ResNet-18 (CIFAR-style stem) for 80x80 cat breeds (8 classes)
   - 3x3 stride=1 stem, no maxpool (better for small images)
   - Augmentations: RandomResizedCrop, HFlip, small Rotation, ColorJitter
   - Normalization to [-1,1]
   - CrossEntropy with label_smoothing
   - AdamW + CosineAnnealingLR
   - No external weights (weights=None)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])  # scale to ~[-1,1]

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.RandomResizedCrop(80, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
        return transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            normalize,
        ])

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class CIFARStemResNet18(nn.Module):
    """
    ResNet-18 with CIFAR-like 3x3 stem for small inputs:
      - conv1: 3x3, stride=1, padding=1
      - no initial maxpool
    """
    def __init__(self, num_classes=8):
        super().__init__()
        # Start from standard ResNet-18 (random init, no external assets)
        self.backbone = models.resnet18(weights=None, num_classes=num_classes)

        # Replace the ImageNet stem with a CIFAR-style stem
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                        padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # remove early downsample

        # Optionally keep everything else the same (avgpool + fc)

    def forward(self, x):
        return self.backbone(x)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CIFARStemResNet18(num_classes=8)

    def forward(self, x):
        return self.model(x)

# Instantiate the network
net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
# Label smoothing can reduce overconfidence / single-class collapse
loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW works well with cosine schedule
optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=2e-4)

############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################
def weights_init(m):
    # He/Kaiming init for conv/linear; BN to 1/0
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# Smooth decay across the whole run
# (a3main.py calls scheduler.step() once per epoch)
epochs = 30  # define early so we can use it below
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8   # keep; you're using a separate validation.py too
batch_size = 64         # reduce if CPU RAM is tight; increase if training is slow but stable
# epochs moved above to share with scheduler; keep here for a3main importers
# (a3main.py reads this after import)
# epochs = 30
