#!/usr/bin/env python3
"""
visualize_activations.py — sample hidden-unit visualizations for each residual block.

Usage:
  python visualize_activations.py --img path/to/image.png --weights savedModel.pth --topk 16

Outputs a set of PNGs:
  activations_stem.png
  activations_layer1_block1.png
  ...
  activations_layer4_block2.png
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

import student                # your model & transforms
from config import device     # your device selection (CPU/GPU)


def build_model(weights_path: str | None) -> torch.nn.Module:
    net = student.Network().to(device)
    net.eval()
    if weights_path and os.path.isfile(weights_path):
        sd = torch.load(weights_path, map_location=device)
        # Accept both full state_dict and whole checkpoints
        if isinstance(sd, dict) and "state_dict" in sd:
            net.load_state_dict(sd["state_dict"], strict=False)
        else:
            net.load_state_dict(sd, strict=False)
        print(f"Loaded weights from: {weights_path}")
    else:
        if weights_path:
            print(f"[WARN] weights file not found: {weights_path} (visualizing random init)")
    return net


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = student.transform("test")  # match eval preprocessing
    x = tfm(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return x


def register_block_hooks(net: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    Register forward hooks to capture post-ReLU outputs of:
      - stem
      - every BasicBlock in layer1..layer4 (2 blocks each)
    Assumes student.Network().model is the ResNet18SE wrapper.
    """
    feats: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    # Locate the inner model (as defined in student.py)
    model = getattr(net, "model", net)

    def save_activation(name):
        def hook(_, __, output):
            feats[name] = output.detach()
        return hook

    # stem
    handles.append(model.stem.register_forward_hook(save_activation("stem")))

    # Layers: each is nn.Sequential of 2 BasicBlockSE
    for li, layer_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=1):
        layer = getattr(model, layer_name)
        for bi, block in enumerate(layer):
            name = f"{layer_name}_block{bi+1}"
            handles.append(block.register_forward_hook(save_activation(name)))

    return feats, handles


def tensor_to_grid(imgs: torch.Tensor, nrow: int = 4, pad: int = 2) -> torch.Tensor:
    """
    imgs: [N, 1, H, W] or [N, H, W] normalized to [0,1]
    Returns a grid tensor [3, H*, W*] suitable for plt.imshow after permute.
    """
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(1)
    grid = make_grid(imgs, nrow=nrow, padding=pad, normalize=False)
    # make_grid outputs [C,H,W]; if single-channel, it stays 1-channel; expand to 3 for nicer viewing
    if grid.size(0) == 1:
        grid = grid.repeat(3, 1, 1)
    return grid


def visualize_activation_map(act: torch.Tensor, topk: int = 16, upscale_to: int | None = None) -> torch.Tensor:
    """
    act: [1, C, H, W]
    Returns a grid image tensor [3, H*, W*] with top-k channels (by mean activation).
    """
    assert act.dim() == 4 and act.size(0) == 1, "Expected [1,C,H,W]"
    _, C, H, W = act.shape

    # Select top-k channels by spatial mean activation
    means = act.view(C, -1).mean(dim=1)
    k = min(topk, C)
    top_idx = torch.topk(means, k=k, largest=True).indices

    # Extract and normalize each map to [0,1] for display
    maps = act[0, top_idx, :, :]  # [k, H, W]
    # Avoid NaNs if flat:
    maps = maps - maps.amin(dim=(1,2), keepdim=True)
    denom = maps.amax(dim=(1,2), keepdim=True).clamp(min=1e-6)
    maps = maps / denom

    # Optional upscaling for readability
    if upscale_to is not None and (H != upscale_to or W != upscale_to):
        maps = F.interpolate(maps.unsqueeze(1), size=(upscale_to, upscale_to), mode="bilinear", align_corners=False).squeeze(1)

    grid = tensor_to_grid(maps, nrow=int(k**0.5) or 4, pad=2)  # roughly square grid
    return grid  # [3, Hgrid, Wgrid]


def save_grid(grid: torch.Tensor, out_path: str):
    """
    grid: [3,H,W] in [0,1]
    """
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid_np)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to a sample image (RGB)")
    ap.add_argument("--weights", default=None, help="Path to savedModel.pth (optional)")
    ap.add_argument("--topk", type=int, default=16, help="Top-k channels to visualize per layer")
    ap.add_argument("--upsample", type=int, default=128, help="Upscale each feature map for readability (pixels)")
    ap.add_argument("--outdir", default="activations", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    net = build_model(args.weights)
    x = load_image(args.img)

    feats, handles = register_block_hooks(net)

    with torch.no_grad():
        _ = net(x)  # one forward pass triggers hooks

    # Clean up hooks ASAP
    for h in handles:
        h.remove()

    # Visualize each captured activation
    # Order them consistently
    ordered_names = ["stem",
                     "layer1_block1", "layer1_block2",
                     "layer2_block1", "layer2_block2",
                     "layer3_block1", "layer3_block2",
                     "layer4_block1", "layer4_block2"]

    for name in ordered_names:
        if name not in feats:
            print(f"[WARN] missing activation for {name} (skipped)")
            continue
        act = feats[name]  # [1,C,H,W]
        grid = visualize_activation_map(act, topk=args.topk, upscale_to=args.upsample)
        save_grid(grid, os.path.join(args.outdir, f"activations_{name}.png"))

    print("Done.")


if __name__ == "__main__":
    main()
