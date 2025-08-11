#!/usr/bin/env python3
"""
student.py — ResNet18-SE (width=1.0) tuned for 80×80 images, AdamW + lighter augments

Changes vs previous:
- Train on full set: train_val_split = 1 (use validation.py on your 400 held-out images)
- Optimizer: AdamW (lr=3e-4, weight_decay=5e-4) for smoother CPU training
- Label smoothing: 0.05 (down from 0.10)
- RandAugment magnitude: 6 (down from 8) + keep RandomErasing(p=0.25)
- Model: ResNet-18 with SE in every block, small-image stem (3×3, stride=2), no pretrained weights
- Scheduler: CosineAnnealingLR across all epochs

Compliant: only torch/torchvision + std libs; model < 50MB.
"""

from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# =========================
# Transforms (80x80 native)
# =========================
IMG_SIZE = 80

def transform(mode):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])  # scale to ~[-1,1]
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=6),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.RandomGrayscale(p=0.10),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])

# =========================
# Model: ResNet18 with SE
# =========================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SEModule(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s

class BasicBlockSE(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, use_se=True):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2   = nn.BatchNorm2d(planes)
        self.se    = SEModule(planes) if use_se else nn.Identity()

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet18SE(nn.Module):
    def __init__(self, num_classes=8, drop_p=0.30):
        super().__init__()
        # Standard ResNet-18 widths
        c1, c2, c3, c4 = 64, 128, 256, 512

        # Small-image stem: 3x3 stride=2 (80 -> 40), no initial maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Stages [2,2,2,2] with strides [1,2,2,2] → 40→20→10→5
        self.layer1 = self._make_layer(BasicBlockSE, in_planes=c1, planes=c1, blocks=2, stride=1, use_se=True)
        self.layer2 = self._make_layer(BasicBlockSE, in_planes=c1, planes=c2, blocks=2, stride=2, use_se=True)
        self.layer3 = self._make_layer(BasicBlockSE, in_planes=c2, planes=c3, blocks=2, stride=2, use_se=True)
        self.layer4 = self._make_layer(BasicBlockSE, in_planes=c3, planes=c4, blocks=2, stride=2, use_se=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(c4, num_classes)

    def _make_layer(self, block, in_planes, planes, blocks: int, stride: int, use_se: bool):
        layers: List[nn.Module] = []
        layers.append(block(in_planes, planes, stride=stride, use_se=use_se))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, stride=1, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet18SE(num_classes=8, drop_p=0.30)
    def forward(self, x):
        return self.model(x)

# Instantiate model
net = Network()

# =========================
# Loss / Optimizer / Scheduler
# =========================
# Lighter label smoothing to avoid underfitting
loss_func = nn.CrossEntropyLoss(label_smoothing=0.05)

# AdamW tends to be smoother/stabler on CPU with strong augments
optimizer = optim.AdamW(
    net.parameters(),
    lr=3e-4,
    weight_decay=5e-4
)

# Kaiming init for conv/linear; BN gamma=1, beta=0
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

# Cosine anneal over full training; a3main.py calls scheduler.step() each epoch
epochs = 80
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# =========================
# Training meta
# =========================
dataset = "./data"
train_val_split = 1        # <— train on ALL data; use validation.py for held-out eval
batch_size = 256           # drop to 192/128 if RAM is tight
# epochs defined above
