"""Load a class-folder dataset, preprocess it and turn it into a feature matrix."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageOps

from src.features import ImageSample, build_feature_matrix
from src.preprocessing.domain import PreprocessingSettings
from src.preprocessing.morphology import create_leaf_mask
from src.preprocessing.pipeline import preprocess_images

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
META_COLUMNS = ["sample_id", "label", "image_sha256"]

# Every variant removes the background. Settings follow preprocessing_demo.ipynb.
BASELINE = "background_only"
VARIANTS = {
    BASELINE: PreprocessingSettings(
        histogram_equalization=False, apply_filter=False, normalize=False,
        remove_background=True,
    ),
    "normalized": PreprocessingSettings(
        histogram_equalization=False, apply_filter=False, remove_background=True
    ),
    "filtered_clahe": PreprocessingSettings(
        histogram_equalization=True, filter_type="gaussian", remove_background=True
    ),
}

FEATURE_GROUPS = ("color", "glcm", "lbp", "shape")  # `lesions` is skipped: nothing segments lesions


def list_images(data_dir, folder_to_label):
    """Return (paths, labels) for every image in the class folders."""
    data_dir = Path(data_dir)
    paths, labels = [], []
    for folder, label in folder_to_label.items():
        if not (data_dir / folder).is_dir():
            raise FileNotFoundError(f"Missing folder: {data_dir / folder}")
        for path in sorted((data_dir / folder).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
                labels.append(label)
    return paths, np.array(labels)


def load_and_preprocess(path, settings):
    """Return (original image, preprocessed array, leaf mask).

    The mask is computed on the original, like the pipeline's `leaf_mask_stage="start"`,
    so it lines up with the preprocessed array.
    """
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    mask = np.asarray(create_leaf_mask(image)) > 0
    return image, np.asarray(preprocess_images([image], settings)[0]), mask


def _image_features(path, label, sample_id, settings, groups):
    _, array, mask = load_and_preprocess(path, settings)
    X, meta = build_feature_matrix([ImageSample(sample_id, array, int(label), mask)], groups=groups)
    return pd.concat([meta, X], axis=1)


def compute_features(data_dir, folder_to_label, settings, groups=FEATURE_GROUPS, n_jobs=-1):
    """Return one DataFrame: META_COLUMNS + features for every image."""
    data_dir = Path(data_dir)
    paths, labels = list_images(data_dir, folder_to_label)
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_image_features)(
            path, label, path.relative_to(data_dir).as_posix(), settings, groups
        )
        for path, label in zip(paths, labels, strict=True)
    )
    return pd.concat(rows, ignore_index=True)


def load_or_compute_features(
    data_dir, folder_to_label, settings, groups=FEATURE_GROUPS, cache_dir="outputs/features"
):
    """`compute_features` cached as CSV, keyed by data, settings and groups."""
    paths, _ = list_images(data_dir, folder_to_label)
    key = repr(
        (str(data_dir), sorted(folder_to_label.items()), settings, tuple(groups), len(paths))
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    cache_file = Path(cache_dir) / f"{Path(data_dir).name.replace(' ', '_')}_{digest}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file)
    df = compute_features(data_dir, folder_to_label, settings, groups)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def load_frames(data_dir, folder_to_label):
    """Features of every variant in VARIANTS, checked to have the same rows."""
    frames = {
        name: load_or_compute_features(data_dir, folder_to_label, settings)
        for name, settings in VARIANTS.items()
    }
    for name, df in frames.items():
        if not df[["sample_id", "label"]].equals(frames[BASELINE][["sample_id", "label"]]):
            raise ValueError(f"{name}: rows differ from {BASELINE}")
        if not np.isfinite(split_features(df)[0].to_numpy()).all():
            raise ValueError(f"{name}: NaN or inf in features")
    return frames


def split_features(df):
    """Return (X, meta): feature columns and META_COLUMNS."""
    return df.drop(columns=META_COLUMNS), df[META_COLUMNS]
