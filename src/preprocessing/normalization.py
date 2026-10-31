import numpy as np
from PIL import Image


def apply_normalization(image: Image.Image, normalization_method: str) -> np.ndarray:
    """Return float32 pixels using global min-max or z-score normalization.

    One shared scale preserves relationships between RGB channels.
    Constant images become zeros.
    """
    pixels = np.asarray(image, dtype=np.float32)
    if normalization_method == "min-max":
        minimum = pixels.min()
        scale = pixels.max() - minimum
        return (pixels - minimum) / scale if scale > 0 else np.zeros_like(pixels)
    if normalization_method == "z-score":
        scale = pixels.std()
        return (pixels - pixels.mean()) / scale if scale > 0 else np.zeros_like(pixels)
    raise ValueError(f"Unknown normalization method: {normalization_method}")
