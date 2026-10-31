import numpy as np
from PIL import Image, ImageFilter


def filter_image(image: Image.Image, filter_type: str) -> Image.Image:
    """Apply a selected filter using a small default kernel."""
    if filter_type == "gaussian":
        return apply_gaussian_filter(image, sigma=0.5)
    if filter_type == "median":
        return apply_median_filter(image, size=3)
    if filter_type == "mean":
        return apply_mean_filter(image, size=3)
    if filter_type == "sharpen":
        return apply_sharpen_filter(image)
    if filter_type == "blur":
        return apply_blur_filter(image)
    raise ValueError(f"Unknown filter type: {filter_type}")


def apply_gaussian_filter(image: Image.Image, sigma: float) -> Image.Image:
    """Smooth noise with a Gaussian kernel."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Sigma must be finite and non-negative.")
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_median_filter(image: Image.Image, size: int) -> Image.Image:
    """Remove impulse noise using an odd-sized neighborhood."""
    if not isinstance(size, int) or size < 1 or size % 2 == 0:
        raise ValueError("Median filter size must be a positive odd integer.")
    return image.filter(ImageFilter.MedianFilter(size=size))


def apply_mean_filter(image: Image.Image, size: int) -> Image.Image:
    """Average pixels in an odd-sized square neighborhood."""
    if not isinstance(size, int) or size < 1 or size % 2 == 0:
        raise ValueError("Mean filter size must be a positive odd integer.")
    return image.filter(ImageFilter.BoxBlur(radius=size // 2))


def apply_sharpen_filter(image: Image.Image) -> Image.Image:
    """Sharpen leaf edges and texture."""
    return image.filter(ImageFilter.SHARPEN)


def apply_blur_filter(image: Image.Image) -> Image.Image:
    """Apply Pillow's standard blur filter."""
    return image.filter(ImageFilter.BLUR)
