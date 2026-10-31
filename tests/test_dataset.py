import numpy as np
import pytest
from PIL import Image

from src.features import FEATURE_GROUPS
from src.features.dataset import ImageSample, build_feature_matrix, load_folder


def test_matrix_keeps_feature_rows_aligned_with_metadata():
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    samples = [
        ImageSample("second", image, "healthy"),
        ImageSample("first", image.copy(), "diseased"),
    ]

    matrix, metadata = build_feature_matrix(iter(samples))

    assert matrix.shape == (2, 46)
    assert list(matrix.index) == list(metadata.index) == [0, 1]
    assert list(metadata.columns) == ["sample_id", "label", "image_sha256"]
    assert metadata["sample_id"].tolist() == ["second", "first"]
    assert metadata["label"].tolist() == ["healthy", "diseased"]
    assert set(metadata.columns).isdisjoint(matrix.columns)
    assert metadata["image_sha256"].nunique() == 1
    assert len(metadata.loc[0, "image_sha256"]) == 64
    assert np.isfinite(matrix.to_numpy()).all()


def test_matrix_supports_all_groups_with_explicit_masks():
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    leaf_mask = np.zeros((32, 32), dtype=bool)
    leaf_mask[4:28, 4:28] = True
    lesion_mask = np.zeros_like(leaf_mask)
    sample = ImageSample("leaf", image, "healthy", leaf_mask, lesion_mask)

    matrix, metadata = build_feature_matrix([sample], groups=FEATURE_GROUPS)

    assert matrix.shape == (1, 54)
    assert metadata.loc[0, "label"] == "healthy"
    assert np.isfinite(matrix.to_numpy()).all()


def test_matrix_rejects_duplicate_sample_ids():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    samples = [
        ImageSample("same", image, "healthy"),
        ImageSample("same", image.copy(), "diseased"),
    ]

    with pytest.raises(ValueError):
        build_feature_matrix(samples)


def test_matrix_rejects_empty_input():
    with pytest.raises(ValueError):
        build_feature_matrix(iter(()))


def test_matrix_preserves_numeric_labels_from_image_loader():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    samples = [
        ImageSample("healthy", image, np.int64(0)),
        ImageSample("ill", image, np.int64(1)),
    ]

    X, metadata = build_feature_matrix(samples)

    assert len(X) == 2
    np.testing.assert_array_equal(metadata.label.to_numpy(), [0, 1])
    assert np.issubdtype(metadata.label.dtype, np.integer)


def test_preprocessing_dry_run_integrates_without_loading_models():
    from src.enhancement import restore_images
    from src.features import ImageSample, build_feature_matrix

    originals = [Image.new("L", (32, 32), 128), Image.new("RGBA", (32, 32))]
    restored = restore_images(originals, dry_run=True)
    samples = [
        ImageSample(str(index), np.asarray(image.convert("RGB")), index)
        for index, image in enumerate(restored)
    ]

    X, metadata = build_feature_matrix(samples)

    assert X.shape == (2, 46)
    assert metadata.label.tolist() == [0, 1]
    assert [image.mode for image in originals] == ["L", "RGBA"]


def test_folder_loader_preserves_rgb_size_and_deterministic_order(tmp_path):
    root = tmp_path / "images"
    (root / "healthy").mkdir(parents=True)
    (root / "diseased").mkdir()
    image = np.zeros((24, 40, 3), dtype=np.uint8)
    image[:, :, 0] = 180
    Image.fromarray(image).save(root / "healthy" / "z.png")
    Image.fromarray(image).save(root / "healthy" / "a.png")
    Image.fromarray(image).save(root / "diseased" / "leaf.png")

    samples = list(load_folder(root))

    assert [sample.sample_id for sample in samples] == [
        "diseased/leaf.png",
        "healthy/a.png",
        "healthy/z.png",
    ]
    assert [sample.label for sample in samples] == ["diseased", "healthy", "healthy"]
    for sample in samples:
        np.testing.assert_array_equal(sample.image, image)
        assert sample.leaf_mask is None
        assert sample.lesion_mask is None


def test_folder_masks_follow_class_and_stem_convention(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    lesions = tmp_path / "lesions"
    for root in (images, masks, lesions):
        (root / "healthy").mkdir(parents=True)
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    leaf_mask = np.zeros((32, 32), dtype=np.uint8)
    leaf_mask[4:28, 4:28] = 255
    lesion_mask = np.zeros_like(leaf_mask)
    Image.fromarray(image).save(images / "healthy" / "leaf.jpg")
    Image.fromarray(leaf_mask).save(masks / "healthy" / "leaf.png")
    Image.fromarray(lesion_mask).save(lesions / "healthy" / "leaf.png")

    samples = list(load_folder(images, masks=masks, lesions=lesions))
    matrix, metadata = build_feature_matrix(samples, groups=FEATURE_GROUPS)

    assert len(samples) == 1
    assert samples[0].leaf_mask.dtype == np.bool_
    assert samples[0].lesion_mask.dtype == np.bool_
    np.testing.assert_array_equal(samples[0].leaf_mask, leaf_mask > 0)
    np.testing.assert_array_equal(samples[0].lesion_mask, lesion_mask > 0)
    assert matrix.shape == (1, 54)
    assert metadata.loc[0, "sample_id"] == "healthy/leaf.jpg"


def test_folder_loader_does_not_silently_replace_missing_masks(tmp_path):
    images = tmp_path / "images"
    (images / "healthy").mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
        images / "healthy" / "leaf.jpg"
    )

    with pytest.raises((FileNotFoundError, ValueError)):
        list(load_folder(images, masks=tmp_path / "missing"))
