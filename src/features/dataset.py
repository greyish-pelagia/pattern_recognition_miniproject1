"""Adapt image samples from a preprocessing pipeline or a class-folder dataset."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .extract import DEFAULT_GROUPS, extract_features

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageSample:
    sample_id: str
    image: np.ndarray
    label: str | int | np.integer
    leaf_mask: np.ndarray | None = None
    lesion_mask: np.ndarray | None = None


def build_feature_matrix(samples, *, groups=DEFAULT_GROUPS):
    """Return aligned (X, sample metadata); IDs and targets never enter X."""
    rows, records = [], []
    seen = set()
    for sample in samples:
        if (
            not isinstance(sample.sample_id, str)
            or not sample.sample_id
            or sample.sample_id in seen
        ):
            raise ValueError("Each sample_id must be a nonempty unique string")
        if (
            isinstance(sample.label, (bool, np.bool_))
            or not isinstance(sample.label, (str, int, np.integer))
            or sample.label == ""
        ):
            raise ValueError("Each sample label must be a nonempty string or integer")
        seen.add(sample.sample_id)
        try:
            rows.append(
                extract_features(
                    sample.image, sample.leaf_mask, sample.lesion_mask, groups=groups
                )
            )
        except ValueError as error:
            raise ValueError(f"{sample.sample_id}: {error}") from error

        # Let training identify exact pixel duplicates when constructing splits.
        image = np.ascontiguousarray(sample.image)
        digest = sha256(
            str((image.shape, image.dtype.str)).encode() + image.tobytes()
        ).hexdigest()
        records.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "image_sha256": digest,
            }
        )
    if not rows:
        raise ValueError("Dataset contains no image samples")
    return pd.DataFrame(rows), pd.DataFrame(records)


def load_folder(root, *, masks=None, lesions=None):
    """Read root/class/image files; optional masks mirror paths with .png suffix.

    No resizing, color correction or segmentation is performed here. Mask PNGs
    must use 0/1 or 0/255. The pipeline can bypass this adapter with ImageSample.
    """
    root = Path(root)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(root)
            with Image.open(path) as source:
                image = np.array(source.convert("RGB"))
            loaded_masks = []
            for mask_root in (masks, lesions):
                if mask_root is None:
                    loaded_masks.append(None)
                    continue
                with Image.open(
                    Path(mask_root) / relative.with_suffix(".png")
                ) as source:
                    mask = np.array(source.convert("L"))
                if not np.isin(mask, (0, 1, 255)).all():
                    raise ValueError(
                        f"Mask for {relative} must be binary (0/1 or 0/255)"
                    )
                loaded_masks.append(mask > 0)
            yield ImageSample(relative.as_posix(), image, folder.name, *loaded_masks)
