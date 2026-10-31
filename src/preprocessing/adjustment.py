import cv2
import numpy as np
from PIL import Image, ImageEnhance


def _auto_adjust(
    image: Image.Image, reference: float, contrast: bool, mask: Image.Image | None
) -> Image.Image:
    pixels = np.asarray(image if image.mode == "L" else image.convert("RGB"), dtype=np.float32)
    luminance = pixels if pixels.ndim == 2 else cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    selected = np.ones(luminance.shape, dtype=bool)
    if mask is not None:
        if mask.mode != "L" or mask.size != image.size:
            raise ValueError("Mask must be an L image with the same size as the input.")
        selected = np.asarray(mask) > 0
    if not selected.any():
        return image.copy()
    mean = luminance[selected].mean()
    if contrast:
        std = luminance[selected].std()
        if std == 0:
            return image.copy()
        adjusted = (luminance - mean) * (reference / std) + mean
    else:
        adjusted = luminance + reference - mean
    if pixels.ndim == 3:
        adjusted = pixels + (adjusted - luminance)[:, :, None]
    result = Image.fromarray(np.clip(np.rint(adjusted), 0, 255).astype(np.uint8))
    if mask is not None:
        result = Image.composite(result, image, mask)
    return result


def adjust_brightness(
    image: Image.Image,
    factor: float | str,
    reference: float = 128.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Use a manual factor or 'auto' to match mean luminance to reference.

    Auto adjusts Y in YCbCr (L for grayscale). An optional mask limits the
    measurement and adjustment. Clipping and rounding limit accuracy.
    """
    if factor == "auto":
        if not np.isfinite(reference) or not 0 <= reference <= 255:
            raise ValueError("Brightness reference must be finite and between 0 and 255.")
        return _auto_adjust(image, reference, False, mask)
    if isinstance(factor, str):
        raise ValueError("Brightness factor must be a number or 'auto'.")
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Brightness factor must be finite and non-negative.")
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(
    image: Image.Image,
    factor: float | str,
    reference: float = 40.0,
    mask: Image.Image | None = None,
) -> Image.Image:
    """Use a manual factor or 'auto' to match luminance standard deviation.

    Auto scales around the current mean; constant images stay unchanged.
    Reference uses the 0–255 luminance scale. Clipping can change the mean.
    """
    if factor == "auto":
        if not np.isfinite(reference) or not 0 <= reference <= 127.5:
            raise ValueError("Contrast reference must be finite and between 0 and 127.5.")
        return _auto_adjust(image, reference, True, mask)
    if isinstance(factor, str):
        raise ValueError("Contrast factor must be a number or 'auto'.")
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Contrast factor must be finite and non-negative.")
    return ImageEnhance.Contrast(image).enhance(factor)


def histogram_equalization(image: Image.Image) -> Image.Image:
    """Apply CLAHE to luminance, preserving leaf and lesion colors."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if image.mode == "L":
        return Image.fromarray(clahe.apply(np.asarray(image)))
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
