"""Checkpoint loading and PIL/tensor conversion for restoration."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate


def load_state_dict(ckpt_path, location='cpu'):
    if Path(ckpt_path).suffix.lower() == '.safetensors':
        from safetensors.torch import load_file
        weights = load_file(ckpt_path, device=location)
    else:
        weights = torch.load(ckpt_path, map_location=location, weights_only=True)
    return weights.get('state_dict', weights)


def PIL2Tensor(img, min_size=1024):
    """Convert RGB PIL to CHW float32 in [-1, 1], aligned to 64 pixels."""
    width, height = img.size
    scale = max(1, min_size / min(width, height))
    aligned_width = max(64, round(width * scale / 64) * 64)
    aligned_height = max(64, round(height * scale / 64) * 64)
    resized = img.resize((aligned_width, aligned_height), Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(pixels).permute(2, 0, 1), height, width


def Tensor2PIL(tensor, height, width):
    """Convert CHW in [-1, 1] to RGB PIL at the requested output dimensions."""
    resized = interpolate(tensor.float().unsqueeze(0), size=(height, width), mode='bicubic')
    pixels = resized.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray((pixels * 127.5 + 127.5).clip(0, 255).astype(np.uint8))
