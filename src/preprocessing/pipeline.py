import numpy as np
from PIL import Image, ImageOps

from src.preprocessing.adjustment import (
    adjust_brightness,
    adjust_contrast,
    histogram_equalization,
)
from src.preprocessing.binarization import apply_binarization
from src.preprocessing.domain import PreprocessingSettings
from src.preprocessing.filtering import filter_image
from src.preprocessing.morphology import (
    apply_morphology,
    create_leaf_mask,
    remove_background,
)
from src.preprocessing.normalization import apply_normalization


def preprocess_images(
    images: list[Image.Image], settings: PreprocessingSettings
) -> list[Image.Image | np.ndarray]:
    """Process RGB copies without changing the inputs.

    Normalize last. Results are float32 arrays when normalization is enabled,
    otherwise PIL images. Binarization is optional because it removes texture
    and color information useful for disease classification.

    Compute the background mask at leaf_mask_stage, even if that step is
    disabled. Auto adjustments use the leaf mask only at the start stage;
    later stages measure brightness and contrast over the whole image.
    """
    processed = []
    if settings.apply_morphology and not settings.binarize:
        raise ValueError("Enable binarization before applying morphology.")
    if settings.remove_background and settings.leaf_mask_stage not in (
        "start", "after_adjustment", "after_filtering", "after_equalization"
    ):
        raise ValueError(f"Unknown leaf mask stage: {settings.leaf_mask_stage}")
    for image in images:
        image = ImageOps.exif_transpose(image).convert("RGB")
        leaf_mask = None
        if settings.remove_background and settings.leaf_mask_stage == "start":
            leaf_mask = create_leaf_mask(image)
        if settings.brightness_adjustment != 1.0:
            image = adjust_brightness(
                image, settings.brightness_adjustment, settings.brightness_reference, leaf_mask
            )
        if settings.contrast_adjustment != 1.0:
            image = adjust_contrast(
                image, settings.contrast_adjustment, settings.contrast_reference, leaf_mask
            )
        if settings.remove_background and settings.leaf_mask_stage == "after_adjustment":
            leaf_mask = create_leaf_mask(image)
        if settings.apply_filter:
            image = filter_image(image, settings.filter_type)
        if settings.remove_background and settings.leaf_mask_stage == "after_filtering":
            leaf_mask = create_leaf_mask(image)
        if settings.histogram_equalization:
            image = histogram_equalization(image)
        if settings.remove_background and settings.leaf_mask_stage == "after_equalization":
            leaf_mask = create_leaf_mask(image)
        if settings.binarize:
            image = apply_binarization(image, per_channel=settings.binarize_per_channel)
        if settings.apply_morphology:
            image = apply_morphology(
                image, settings.morphology_operation, settings.morphology_kernel_size
            )
        if leaf_mask is not None:
            image = remove_background(image, leaf_mask)
        if settings.normalize:
            image = apply_normalization(image, settings.normalization_method)
        processed.append(image)
    return processed
