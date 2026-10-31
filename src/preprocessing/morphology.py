import cv2
import numpy as np
from PIL import Image


def apply_morphology(
    image: Image.Image, operation: str = "opening-closing", kernel_size: int = 3
) -> Image.Image:
    """Clean binary L or RGB images; process RGB channels independently."""
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer.")
    pixels = np.asarray(image)
    if image.mode not in ("L", "RGB") or not np.isin(pixels, [0, 255]).all():
        raise ValueError("Provide a binary L or RGB image with values 0 and 255.")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if operation == "opening-closing":
        pixels = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(cv2.morphologyEx(pixels, cv2.MORPH_CLOSE, kernel))
    operations = {
        "opening": cv2.MORPH_OPEN,
        "closing": cv2.MORPH_CLOSE,
        "erosion": cv2.MORPH_ERODE,
        "dilation": cv2.MORPH_DILATE,
    }
    if operation not in operations:
        raise ValueError(f"Unknown morphology operation: {operation}")
    return Image.fromarray(cv2.morphologyEx(pixels, operations[operation], kernel))


def create_leaf_mask(
    image: Image.Image, opening_size: int = 3, closing_size: int = 7
) -> Image.Image:
    """Fill the largest leaf contour's convex hull, including boundary lesions.

    A convex hull closes missing edge regions but can include background in
    natural concavities of the leaf outline.
    """
    rgb = np.asarray(image.convert("RGB"))
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    green = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1]
    green = cv2.GaussianBlur(green, (5, 5), 0)
    _, green_mask = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(mask, green_mask)
    mask = apply_morphology(Image.fromarray(mask), "opening", opening_size)
    mask = apply_morphology(mask, "closing", closing_size)
    contours, _ = cv2.findContours(
        np.asarray(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros(rgb.shape[:2], dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        cv2.drawContours(filled, [hull], -1, 255, cv2.FILLED)
    return Image.fromarray(filled)


def remove_background(image: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """Apply a leaf mask, replacing the background with black pixels."""
    if mask is None:
        mask = create_leaf_mask(image)
    if mask.size != image.size or mask.mode != "L":
        raise ValueError("Mask must be an L image with the same size as the input.")
    return Image.composite(image, Image.new(image.mode, image.size, 0), mask)
