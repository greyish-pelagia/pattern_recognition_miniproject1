import numpy as np
from PIL import Image, ImageDraw

from src.training.data import (
    VARIANTS,
    compute_features,
    list_images,
)
from src.training.experiment import group_train_test_split

CLASSES = {"healthy": 0, "ill": 1}


def _write_dataset(root):
    """Two 96x96 gray-background leaf images per class, with a little noise."""
    for folder, color in (("healthy", (40, 160, 40)), ("ill", (120, 90, 40))):
        (root / folder).mkdir()
        for seed in range(2):
            rng = np.random.default_rng(seed)
            image = Image.new("RGB", (96, 96), (128, 128, 128))
            ImageDraw.Draw(image).ellipse((16, 12, 80, 84), fill=color)
            noisy = np.clip(np.asarray(image) + rng.integers(-8, 9, (96, 96, 3)), 0, 255)
            Image.fromarray(noisy.astype(np.uint8)).save(root / folder / f"{seed}.jpg")


def test_group_split_is_stratified_and_keeps_duplicates_together():
    y = np.array([0, 1] * 50)
    groups = np.arange(100)
    groups[1::10] = groups[0::10][: len(groups[1::10])]  # a few duplicate groups
    train, test = group_train_test_split(y, groups, test_size=0.3, seed=0)
    assert not set(groups[train]) & set(groups[test])
    assert sorted(np.concatenate([train, test])) == list(range(100))
    assert 0.25 <= len(test) / 100 <= 0.35
    assert abs(y[test].mean() - 0.5) <= 0.1


def test_every_variant_has_all_50_features(tmp_path):
    _write_dataset(tmp_path)
    assert len(list_images(tmp_path, CLASSES)[0]) == 4
    for settings in VARIANTS.values():
        df = compute_features(tmp_path, CLASSES, settings, n_jobs=1)
        assert df["label"].tolist() == [0, 0, 1, 1]
        assert df.shape[1] == 3 + 50
        assert not df.isna().any().any()
