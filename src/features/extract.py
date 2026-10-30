"""Measure color, texture and geometry without fitting dataset-level parameters."""

import numpy as np
from skimage import color, feature, measure, morphology, util

FEATURE_GROUPS = ("color", "glcm", "lbp", "shape", "lesions")
DEFAULT_GROUPS = ("color", "glcm", "lbp")
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 2, 4)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPERTIES = ("contrast", "homogeneity", "energy", "correlation")
LBP_SETTINGS = ((1, 8), (2, 16))


def _validate_mask(mask, shape, name):
    mask = np.asarray(mask)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be a bool array with shape {shape}")
    return mask


def extract_features(image, leaf_mask=None, lesion_mask=None, *, groups=DEFAULT_GROUPS):
    """Return named features for RGB uint8 or float [0, 1] input.

    Masks are bool arrays aligned with the image. Geometry requires explicit
    masks; an empty lesion mask means no detected lesions, not missing data.
    With no leaf mask, color and texture describe the entire image.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("image must have nonempty shape (H, W, 3) in RGB order")
    if image.dtype != np.uint8 and not np.issubdtype(image.dtype, np.floating):
        raise ValueError("image must be uint8 or floating point in [0, 1]")
    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    if np.issubdtype(image.dtype, np.floating) and (image.min() < 0 or image.max() > 1):
        raise ValueError("floating point image must be in [0, 1]")

    groups = tuple(groups)
    if (
        not groups
        or len(set(groups)) != len(groups)
        or set(groups) - set(FEATURE_GROUPS)
    ):
        raise ValueError(f"groups must be a nonempty unique subset of {FEATURE_GROUPS}")

    roi = np.ones(image.shape[:2], dtype=bool)
    if leaf_mask is not None:
        roi = _validate_mask(leaf_mask, image.shape[:2], "leaf_mask")
    if not roi.any():
        raise ValueError("leaf_mask is empty")
    if "shape" in groups and leaf_mask is None:
        raise ValueError("shape features require leaf_mask")
    if lesion_mask is not None:
        lesion_mask = _validate_mask(lesion_mask, roi.shape, "lesion_mask")
        if leaf_mask is None or np.any(lesion_mask & ~roi):
            raise ValueError("lesion_mask requires leaf_mask and must be its subset")
    if "lesions" in groups and lesion_mask is None:
        raise ValueError("lesions features require leaf_mask and lesion_mask")

    image = util.img_as_float32(image)
    values = {}
    if "color" in groups:
        pixels = color.rgb2lab(image)[roi]
        for index, channel in enumerate(("L", "a", "b")):
            values[f"color_{channel}_mean"] = float(pixels[:, index].mean())
            values[f"color_{channel}_std"] = float(pixels[:, index].std())

    if "glcm" in groups or "lbp" in groups:
        gray = util.img_as_ubyte(color.rgb2gray(image))
    if "glcm" in groups:
        values.update(_glcm_features(gray, roi))
    if "lbp" in groups:
        values.update(_lbp_features(gray, roi))

    if "shape" in groups:
        leaf = measure.regionprops(roi.astype(np.uint8))[0]
        perimeter = measure.perimeter_crofton(roi, directions=4)
        values.update(
            shape_eccentricity=float(leaf.eccentricity),
            shape_solidity=float(leaf.solidity),
            shape_extent=float(leaf.extent),
            shape_circularity=float(4 * np.pi * leaf.area / perimeter**2),
        )

    if "lesions" in groups:
        # Relative areas avoid making the photographed leaf size a disease cue.
        areas = np.array(
            [r.area for r in measure.regionprops(measure.label(lesion_mask))]
        )
        leaf_area = float(roi.sum())
        values.update(
            lesions_ratio=float(lesion_mask.sum() / leaf_area),
            lesions_count=float(len(areas)),
            lesions_largest_fraction=float(areas.max() / leaf_area)
            if areas.size
            else 0.0,
            lesions_mean_fraction=float(areas.mean() / leaf_area)
            if areas.size
            else 0.0,
        )
    return values


def _glcm_features(gray, roi):
    # Reserve one gray level for background, then discard all pairs containing it.
    quantized = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    quantized[~roi] = GLCM_LEVELS
    matrix = feature.graycomatrix(
        quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS + 1,
        symmetric=True,
    )[:GLCM_LEVELS, :GLCM_LEVELS].astype(float)
    counts = matrix.sum(axis=(0, 1), keepdims=True)
    if np.any(counts == 0):
        raise ValueError("leaf_mask has no valid GLCM pairs for some offsets")
    matrix /= counts

    values = {}
    for prop in GLCM_PROPERTIES:
        direction_means = feature.graycoprops(matrix, prop).mean(axis=1)
        for distance, value in zip(GLCM_DISTANCES, direction_means, strict=True):
            values[f"glcm_d{distance}_{prop}"] = float(value)
    return values


def _lbp_features(gray, roi):
    values = {}
    for radius, points in LBP_SETTINGS:
        # Exclude centers whose neighborhoods touch background or the image edge.
        interior = morphology.erosion(
            roi, footprint=np.ones((2 * radius + 1, 2 * radius + 1)), mode="min"
        )
        if not interior.any():
            raise ValueError(
                f"leaf_mask has no complete LBP neighborhoods at radius {radius}"
            )
        codes = feature.local_binary_pattern(gray, points, radius, method="uniform")
        histogram = np.bincount(codes[interior].astype(int), minlength=points + 2)
        histogram = histogram / histogram.sum()
        for index, value in enumerate(histogram):
            values[f"lbp_r{radius}_{index}"] = float(value)
    return values
