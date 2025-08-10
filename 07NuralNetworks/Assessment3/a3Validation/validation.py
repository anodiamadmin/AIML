#!/usr/bin/env python3
"""
validation.py

Validate savedModel.pth on the images in ./validation
Folder structure expected:
validation/
  0/, 1/, ..., 7/   (class folders with images)

This script:
- Loads student.Network and student.transform('test')
- Restores weights from savedModel.pth
- Evaluates accuracy and confusion matrix on the validation set
"""

import os
import sys
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from config import device
import student

try:
    import sklearn.metrics as metrics
    _USE_SKLEARN = True
except Exception:
    _USE_SKLEARN = False


VAL_DIR = "./validation"
MODEL_PATH = "./savedModel.pth"


def load_validation_loader(batch_size):
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

    dataset = torchvision.datasets.ImageFolder(
        root=VAL_DIR,
        transform=student.transform('test')
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No validation images found under {VAL_DIR}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader


def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run `python3 a3main.py` first to train and save the model."
        )
    net = student.Network().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def evaluate(net, loader, num_classes=8):
    total_images = 0
    total_correct = 0
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            preds = outputs.argmax(dim=1)

            total_images += labels.size(0)
            total_correct += (preds == labels).sum().item()

            if _USE_SKLEARN:
                conf_matrix += metrics.confusion_matrix(
                    labels.cpu(), preds.cpu(), labels=list(range(num_classes))
                )
            else:
                # Fallback confusion matrix without sklearn
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    conf_matrix[int(t.item()), int(p.item())] += 1

    accuracy = 100.0 * total_correct / max(total_images, 1)
    return accuracy, conf_matrix, total_images


def print_report(accuracy, conf_matrix, total_images, class_names):
    print(f"Validated on {total_images} images")
    print(f"Overall accuracy: {accuracy:.2f}%\n")

    np.set_printoptions(precision=2, suppress=True)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(conf_matrix)
    print()

    # Per-class accuracy
    print("Per-class accuracy:")
    for i, name in enumerate(class_names):
        true_count = conf_matrix[i].sum()
        correct = conf_matrix[i, i]
        cls_acc = 100.0 * correct / true_count if true_count > 0 else 0.0
        print(f"  Class {name}: {cls_acc:.2f}%  ({correct}/{true_count})")


def main():
    # Load data
    val_dataset, val_loader = load_validation_loader(batch_size=student.batch_size)
    class_names = [str(k) for k in range(8)]
    # If folders are '0'..'7', ImageFolder.classes will be strings '0'..'7'
    # but we print our own stable order 0..7 to match labels used in training.
    # (You can replace class_names with val_dataset.classes if you prefer.)
    # class_names = val_dataset.classes

    # Load model
    net = load_model()

    # Evaluate
    accuracy, conf_matrix, total_images = evaluate(net, val_loader, num_classes=8)

    # Report
    print_report(accuracy, conf_matrix, total_images, class_names)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[validation.py] Error: {e}", file=sys.stderr)
        sys.exit(1)
