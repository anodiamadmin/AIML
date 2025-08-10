"""
   cnn.py

   UNSW ZZEN9444 Neural Networks and Deep Learning

ResNet-18 version (no external assets / no pretrained weights):
- Backbone: torchvision.models.resnet18(weights=None) with num_classes=8.
- Small-input tweak: keep the 7x7 stem but REMOVE the initial maxpool to retain more
  spatial detail for 80x80 images (improves early signal for small inputs).
- Augmentation: mild random crop/flip/rotation; normalize to [-1,1].
- Loss/Optim: CrossEntropy + Adam (lr=1e-3, weight_decay=1e-4).
- Scheduler: StepLR every 8 epochs (gamma=0.5).
- Size: ~11.7M params (≈47MB in FP32), fits under 50MB limit.

This file is backend-agnostic; a3main.py/config.py handle device selection.
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
    """
    Different transforms for training and testing. Images are 80x80 RGB.
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])  # scale to ~[-1,1]

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.RandomResizedCrop(80, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
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
        return transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            normalize,
        ])


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a vanilla ResNet-18 with random initialization (no external weights)
        self.backbone = models.resnet18(weights=None, num_classes=8)

        # Small-input tweak: disable the initial 3x downsampling from maxpool
        # (keep 7x7 conv stride=2, but removing maxpool preserves more detail for 80x80).
        # For even more detail, you could also set stride=1, but we keep stride=2 to remain close to stock.
        self.backbone.maxpool = nn.Identity()

        # Optional: Slightly stronger regularization just before the classifier
        # by adding dropout on the pooled features (inserted via forward)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Copy of torchvision ResNet forward, but we slip in dropout before fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        # self.backbone.maxpool is Identity (no-op)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.fc(x)  # 8 classes
        return x


# Instantiate the network
net = Network()


############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################
def weights_init(m):
    """
    He/Kaiming init for Conv/Linear; BatchNorm to 1/0.
    a3main.py calls net.apply(weights_init) once at start.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # ResNet fc follows ReLU features; Kaiming is fine here too
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    # Skip other modules (Identities, Pooling, etc.)

# Halve LR every 8 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)


############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
# ResNet-18 is heavier than the earlier CNN; keep batch size moderate for CPU
batch_size = 32     #64
epochs = 15         #25