import cv2
import numpy as np
from PIL import Image


def binarize_image(image: Image.Image, threshold: int) -> Image.Image:
    """Return an L image with 255 for pixels above the threshold, else 0."""
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")
    pixels = np.asarray(image.convert("L"))
    return Image.fromarray(np.where(pixels > threshold, 255, 0).astype(np.uint8))


def binarize_image_per_channel(
    image: Image.Image, thresholds: tuple[int, int, int]
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Threshold R, G and B independently, returning three L images."""
    if len(thresholds) != 3:
        raise ValueError("Provide exactly three RGB thresholds.")
    red, green, blue = image.convert("RGB").split()
    return (
        binarize_image(red, thresholds[0]),
        binarize_image(green, thresholds[1]),
        binarize_image(blue, thresholds[2]),
    )


def combine_binarized_channels(
    binarized_channels: tuple[Image.Image, Image.Image, Image.Image],
) -> Image.Image:
    """Merge three binary L images into one RGB image."""
    return Image.merge("RGB", binarized_channels)


def calculate_otsu_threshold(image: Image.Image) -> int:
    """Calculate Otsu's threshold from grayscale pixels."""
    threshold, _ = cv2.threshold(
        np.asarray(image.convert("L")), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return int(threshold)


def calculate_otsu_threshold_per_channel(image: Image.Image) -> tuple[int, int, int]:
    """Calculate a separate Otsu threshold for R, G and B."""
    red, green, blue = image.convert("RGB").split()
    return (
        calculate_otsu_threshold(red),
        calculate_otsu_threshold(green),
        calculate_otsu_threshold(blue),
    )


def apply_binarization(
    image: Image.Image, threshold: int | None = None, per_channel: bool = False
) -> Image.Image:
    """Use a fixed threshold, or Otsu when threshold is None.

    Per-channel mode returns RGB; grayscale mode returns L.
    """
    if per_channel:
        thresholds = (
            calculate_otsu_threshold_per_channel(image)
            if threshold is None
            else (threshold, threshold, threshold)
        )
        return combine_binarized_channels(binarize_image_per_channel(image, thresholds))
    if threshold is None:
        threshold = calculate_otsu_threshold(image)
    return binarize_image(image, threshold)
