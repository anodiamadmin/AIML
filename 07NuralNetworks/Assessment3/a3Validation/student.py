#!/usr/bin/env python3
"""
student.py — ResNet18-SE (80x80) with Stochastic Depth + tiny TTA (flip)
Compliance with given conditions:
- Uses only approved libraries (torch/torchvision + stdlib)
- No pretrained weights (all layers randomly initialized)
- Model size < 50MB (≈ ResNet-18 width=1.0 with SE; ~47MB FP32 (float32))
- CPU/GPU agnostic (device chosen by config.py and a3main.py)

Core ideas to fight overfitting yet keep accuracy high:
- Strong but stable data augmentation at 80x80 (no upscaling)
- Squeeze-and-Excitation (SE) in each basic block for channel attention
- Stochastic Depth (DropPath) across blocks (0→0.10), regularizes residuals
- Dropout before FC
- Label smoothing, weight decay
- SGD + Nesterov + cosine annealing (good validation behavior with BN)
- Tiny eval-time TTA: average logits of {image, horizontal flip} for a small bump
"""

# ---- PyTorch / TorchVision core
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# =========================
# Transforms (80x80 native) or (96x96) or (128x128)
# =========================
# Keep everything at 80x80 to match dataset’s native resolution.
# IMG_SIZE = 80     # native resolution
# Some accuracy gain is expected as the minute features are magnified.
IMG_SIZE = 96   # Standard Transform
# Considerable accuracy gain is expected as the minute features are magnified.
# Model size will not increase as it is independent of input size.
# Training time will be increased by (128/96)^2 = 16/9 =~ 1.8 times
# IMG_SIZE = 128  # High performance

def transform(mode):
    """
    Return a torchvision transform pipeline.
    - For 'train': use strong but stable augmentation to improve generalization.
    - For 'test': deterministic resize + normalize only (no augmentation).
    """
    # Normalize RGB channels to roughly [-1, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    if mode == 'train':
        return transforms.Compose([
            # Randomly crop/rescale within the original 80x80 canvas
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            # Horizontal flips are natural for cat images (left/right symmetry)
            transforms.RandomHorizontalFlip(p=0.5),
            # RandAugment adds a couple of lightweight perturbations per image
            transforms.RandAugment(num_ops=2, magnitude=5),
            # Mild photometric changes
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            # Occasionally drop color to force reliance on shape/texture
            transforms.RandomGrayscale(p=0.10),
            # Convert PIL image to Tensor (C,H,W) in [0,1]
            transforms.ToTensor(),
            # Normalize to [-1,1]-ish
            normalize,
            # Cutout-style regularization: erase a random patch
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])
    elif mode == 'test':
        return transforms.Compose([
            # Deterministic path: resize to 80x80
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Fallback (not used by a3main.py, but safe)
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


# =========================
# Model: ResNet18 with SE
# =========================

def conv3x3(in_planes, out_planes, stride=1):
    """
    Standard 3x3 conv used in ResNet with padding=1.
    Bias=False since BatchNorm follows and absorbs bias.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation block:
    - Global average pool to get channel statistics
    - Two FC layers with bottleneck (reduction) and sigmoid
    - Scale (recalibrate) channels of the input feature map
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)  # ensure at least 8 units
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)      # Squeeze to (B, C)
        s = self.fc(s).view(b, c, 1, 1)  # Excitation to (B, C, 1, 1)
        return x * s                     # Scale channels


class StochasticDepth(nn.Module):
    """
    DropPath / Stochastic Depth:
    - Randomly drop the residual branch during training with prob p
    - Scale kept paths by 1/(1-p) to preserve expectation
    - Acts per sample, broadcast over (Chanels C, Height H, Width W)
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        # Bernoulli mask per sample, shape (B,1,1,1)
        mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        # Scale to keep expected magnitude the same
        return x * mask / keep


class BasicBlockSE(nn.Module):
    """
    ResNet BasicBlock with optional SE and Stochastic Depth on the residual branch.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> SE -> (DropPath) -> +skip -> ReLU
    BN => Batch Normalization; SE => Squeeze-and-Excitation (from the SENet architecture)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_se=True, drop_path_p=0.0):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2   = nn.BatchNorm2d(planes)
        self.se    = SEModule(planes) if use_se else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_p)

        # If spatial size or channel count changes, project the identity
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)           # Channel attention
        out = self.drop_path(out)    # Maybe drop this residual branch

        if self.downsample is not None:
            identity = self.downsample(x)  # Align for addition

        out = out + identity
        out = self.relu(out)          # Final activation
        return out


class ResNet18SE(nn.Module):
    """
    ResNet-18 (width=1.0) backbone adapted for 80x80 images:
    - Small-image stem: 3x3, stride=2 (128->64), no initial maxpool
    - Stages [2,2,2,2] with strides [1,2,2,2] -> 64->32->16->8 feature maps
    - SE in every block; Stochastic Depth increases linearly across blocks
    - GAP -> Dropout -> FC(8)
    SE => Squeeze-and-Excitation; GAP => Global Average Pooling; FC(8) => Fully Connected layer with 8 outputs
    """
    def __init__(self, num_classes=8, drop_p=0.40, drop_path_max=0.10):
        super().__init__()

        # Standard ResNet-18 channel sizes per stage
        c1, c2, c3, c4 = 64, 128, 256, 512

        # Stem: early downsample to 64x64 while keeping small-kernel detail
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1, bias=False),  # 80 -> 40
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            # Optional (if 128×128 is too slow on CPU): Add a tiny maxpool after the stem to shrink earlier
            # keeps accuracy reasonable while cutting FLOPs
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64->32 (then stages: 32->16->8->4)
        )

        # Prepare a per-block DropPath schedule (8 total blocks in ResNet-18)
        # Linearly ramp from 0.0 to drop_path_max across blocks
        dp = []
        n_blocks = 8
        for i in range(n_blocks):
            dp.append(drop_path_max * float(i) / (n_blocks - 1))  # [0.0, ..., drop_path_max]

        # Stage 1 (stride 1): channels = 64, feature size 40x40
        self.layer1 = nn.Sequential(
            BasicBlockSE(c1, c1, stride=1, use_se=True, drop_path_p=dp[0]),
            BasicBlockSE(c1, c1, stride=1, use_se=True, drop_path_p=dp[1]),
        )
        # Stage 2 (stride 2): channels = 128, feature size 20x20
        self.layer2 = nn.Sequential(
            BasicBlockSE(c1, c2, stride=2, use_se=True, drop_path_p=dp[2]),
            BasicBlockSE(c2, c2, stride=1, use_se=True, drop_path_p=dp[3]),
        )
        # Stage 3 (stride 2): channels = 256, feature size 10x10
        self.layer3 = nn.Sequential(
            BasicBlockSE(c2, c3, stride=2, use_se=True, drop_path_p=dp[4]),
            BasicBlockSE(c3, c3, stride=1, use_se=True, drop_path_p=dp[5]),
        )
        # Stage 4 (stride 2): channels = 512, feature size 5x5
        self.layer4 = nn.Sequential(
            BasicBlockSE(c3, c4, stride=2, use_se=True, drop_path_p=dp[6]),
            BasicBlockSE(c4, c4, stride=1, use_se=True, drop_path_p=dp[7]),
        )

        # Global average pool -> flatten -> dropout -> linear head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(c4, num_classes)

    def forward_single(self, x):
        """
        Single forward path (no TTA).
        Used for both training and as the base path in eval-time TTA.
        TTA => Test-Time Augmentation
        """
        x = self.stem(x)     # 80x80 -> 40x40
        x = self.layer1(x)   # 40x40
        x = self.layer2(x)   # 20x20
        x = self.layer3(x)   # 10x10
        x = self.layer4(x)   # 5x5
        x = self.avgpool(x)  # -> (B, C, 1, 1)
        x = torch.flatten(x, 1)  # -> (B, C)
        x = self.dropout(x)
        return self.fc(x)    # -> (B, 8)

    def forward(self, x):
        """
        During eval (net.eval()), apply a tiny test-time augmentation (TTA):
        average logits of the original and its horizontal flip for a free boost.
        During training (net.train()), use the single path.
        """
        if not self.training:
            logits1 = self.forward_single(x)
            logits2 = self.forward_single(torch.flip(x, dims=[3]))  # flip on width axis
            return (logits1 + logits2) * 0.5
        return self.forward_single(x)


class Network(nn.Module):
    """
    Thin wrapper so a3main.py can instantiate `student.Network()`.
    """
    def __init__(self):
        super().__init__()
        # Dropout=0.40 & DropPath up to 0.10 help curb overfitting past ~30 epochs
        self.model = ResNet18SE(num_classes=8, drop_p=0.40, drop_path_max=0.10)

    def forward(self, x):
        return self.model(x)


# Instantiate model (a3main.py will move it to device and call .train()/.eval())
net = Network()


# =========================
# Loss / Optimizer / Scheduler
# =========================

# CrossEntropy with light label smoothing (0.05) reduces overconfidence
loss_func = nn.CrossEntropyLoss(label_smoothing=0.05)

# Stochastic Gradient Descent + momentum + Nesterov often yields better validation with Batch Normalization than Adam Decoupled Weight Decay
# lr=0.08 works well here with cosine annealing over 50 epochs
optimizer = optim.SGD(
    net.parameters(),
    lr=0.08,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-3,   # decoupled L2-style regularization (Decoupled Weight Decay Regularization)
)

def weights_init(m):
    """
    Custom weight init (a3main.py calls net.apply(weights_init) once):
    - Conv/Linear: Kaiming (He) init for ReLU
    - BatchNorm: gamma=1, beta=0
    """
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

# Cosine annealing across the whole training run.
# NOTE: a3main.py calls scheduler.step() once per epoch.
epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# =========================
# Training meta (read by a3main.py)
# =========================
dataset = "./data"   # training root: folders '0'..'7' under ./data
train_val_split = 0.8  # train on ALL available training images
batch_size = 128     # 256 if RAM permits. Bump down to 192/128 if hit RAM limits
# epochs defined above so scheduler knows T_max
