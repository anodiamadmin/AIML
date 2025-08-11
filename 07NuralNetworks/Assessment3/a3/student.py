#!/usr/bin/env python3
"""
student.py — ResNet18-SE (80x80) with Stochastic Depth + TTA to curb overfitting

- Model: ResNet-18 with SE in every block, small-image stem (3×3, stride=2), no pretrained weights.
- Regularization:
    * Stochastic Depth (DropPath) across blocks (0 -> 0.10)
    * Dropout 0.40 before FC
    * Label smoothing 0.05
    * RandAugment (magnitude=5), ColorJitter, RandomGrayscale, RandomErasing p=0.35
- Eval-time TTA: average logits of {identity, horizontal flip}
- Optimizer: SGD + momentum + Nesterov, weight_decay=1e-3
- Scheduler: CosineAnnealingLR over 50 epochs
- Train on ALL data (train_val_split = 1). Use validation.py for your 400-image holdout.

Compliant: torch/torchvision only; no pretrained weights; model < 50MB.
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
            transforms.RandAugment(num_ops=2, magnitude=5),  # slightly lighter for stability
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            transforms.RandomGrayscale(p=0.10),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
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


class StochasticDepth(nn.Module):
    """DropPath: randomly drop residual branch during training (per-sample)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        # Bernoulli mask per sample, broadcast over channels/H/W
        mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class BasicBlockSE(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_se=True, drop_path_p=0.0):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEModule(planes) if use_se else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_p)

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
        # DropPath on the residual branch (the conv path)
        out = self.drop_path(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet18SE(nn.Module):
    def __init__(self, num_classes=8, drop_p=0.40, drop_path_max=0.10):
        super().__init__()
        # Standard ResNet-18 widths
        c1, c2, c3, c4 = 64, 128, 256, 512

        # Small-image stem: 3x3 stride=2 (80 -> 40), no initial maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Stochastic Depth schedule per block (8 blocks total in [2,2,2,2])
        dp = []
        n_blocks = 8
        for i in range(n_blocks):
            dp.append(drop_path_max * float(i) / (n_blocks - 1))  # 0 -> drop_path_max

        # Build layers with per-block drop rates
        self.layer1 = nn.Sequential(
            BasicBlockSE(c1, c1, stride=1, use_se=True, drop_path_p=dp[0]),
            BasicBlockSE(c1, c1, stride=1, use_se=True, drop_path_p=dp[1]),
        )
        self.layer2 = nn.Sequential(
            BasicBlockSE(c1, c2, stride=2, use_se=True, drop_path_p=dp[2]),
            BasicBlockSE(c2, c2, stride=1, use_se=True, drop_path_p=dp[3]),
        )
        self.layer3 = nn.Sequential(
            BasicBlockSE(c2, c3, stride=2, use_se=True, drop_path_p=dp[4]),
            BasicBlockSE(c3, c3, stride=1, use_se=True, drop_path_p=dp[5]),
        )
        self.layer4 = nn.Sequential(
            BasicBlockSE(c3, c4, stride=2, use_se=True, drop_path_p=dp[6]),
            BasicBlockSE(c4, c4, stride=1, use_se=True, drop_path_p=dp[7]),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(c4, num_classes)

    def forward_single(self, x):
        x = self.stem(x)
        x = self.layer1(x)  # 40x40
        x = self.layer2(x)  # 20x20
        x = self.layer3(x)  # 10x10
        x = self.layer4(x)  # 5x5
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def forward(self, x):
        # During eval, do a tiny TTA: average logits of (x, flipped-x)
        if not self.training:
            logits1 = self.forward_single(x)
            logits2 = self.forward_single(torch.flip(x, dims=[3]))  # horizontal flip (W axis)
            return (logits1 + logits2) * 0.5
        return self.forward_single(x)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet18SE(num_classes=8, drop_p=0.40, drop_path_max=0.10)


    def forward(self, x):
        return self.model(x)

# Instantiate model (no torch.compile to avoid Windows toolchain issues)
net = Network()

# =========================
# Loss / Optimizer / Scheduler
# =========================
# Lighter label smoothing avoids underfitting while still regularizing
loss_func = nn.CrossEntropyLoss(label_smoothing=0.05)

# SGD often gives better test accuracy with BN once training is stable
optimizer = optim.SGD(
    net.parameters(),
    lr=0.08, momentum=0.9, nesterov=True, weight_decay=1e-3
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

# Cosine anneal over full training; a3main.py calls scheduler.step() once/epoch
epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# =========================
# Training meta
# =========================
dataset = "./data"
train_val_split = 1       # train on ALL training images; validate with validation.py
batch_size = 256          # drop to 192/128 if RAM is tight
# epochs defined above
