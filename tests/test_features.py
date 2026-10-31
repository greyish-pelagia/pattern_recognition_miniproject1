import numpy as np
import pytest

from src.features import FEATURE_GROUPS, extract_features


@pytest.fixture
def sample():
    image = np.random.default_rng(12).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    leaf_mask = np.zeros((32, 32), dtype=bool)
    leaf_mask[5:27, 5:27] = True
    lesion_mask = np.zeros_like(leaf_mask)
    lesion_mask[9:11, 9:11] = True
    lesion_mask[20:22, 20:22] = True
    return image, leaf_mask, lesion_mask


def test_background_does_not_change_object_features(sample):
    image, leaf_mask, lesion_mask = sample
    changed = image.copy()
    changed[~leaf_mask] = 255 - changed[~leaf_mask]

    original = extract_features(image, leaf_mask, lesion_mask, groups=FEATURE_GROUPS)
    result = extract_features(changed, leaf_mask, lesion_mask, groups=FEATURE_GROUPS)

    assert result.keys() == original.keys()
    np.testing.assert_allclose(list(result.values()), list(original.values()))


def test_masked_glcm_excludes_background_and_boundary_transitions(sample):
    image, leaf_mask, _ = sample
    image[leaf_mask] = 128

    result = extract_features(image, leaf_mask, groups=("glcm",))

    for name, value in result.items():
        expected = 0 if "contrast" in name else 1
        assert value == pytest.approx(expected), name


def test_lbp_histograms_are_normalized(sample):
    image, leaf_mask, _ = sample
    result = extract_features(image, leaf_mask, groups=("lbp",))

    for radius, bins in ((1, 10), (2, 18)):
        histogram = [result[f"lbp_r{radius}_{index}"] for index in range(bins)]
        assert min(histogram) >= 0
        assert sum(histogram) == pytest.approx(1)


def test_lbp_ignores_neighborhoods_crossing_mask_boundary(sample):
    image, leaf_mask, _ = sample
    image[leaf_mask] = 128
    dark_background = image.copy()
    dark_background[~leaf_mask] = 0
    light_background = image.copy()
    light_background[~leaf_mask] = 255

    dark = extract_features(dark_background, leaf_mask, groups=("lbp",))
    light = extract_features(light_background, leaf_mask, groups=("lbp",))

    assert dark == light
    assert dark["lbp_r1_8"] == pytest.approx(1)
    assert dark["lbp_r2_16"] == pytest.approx(1)


def test_empty_lesion_mask_means_no_detected_lesions(sample):
    image, leaf_mask, lesion_mask = sample
    lesion_mask[:] = False

    result = extract_features(image, leaf_mask, lesion_mask, groups=("lesions",))

    assert result
    assert all(value == 0 for value in result.values())


def test_lesion_measurements_use_leaf_area(sample):
    image, leaf_mask, lesion_mask = sample
    result = extract_features(image, leaf_mask, lesion_mask, groups=("lesions",))

    assert result["lesions_count"] == 2
    assert result["lesions_ratio"] == pytest.approx(8 / leaf_mask.sum())
    assert result["lesions_largest_fraction"] == pytest.approx(4 / leaf_mask.sum())
    assert result["lesions_mean_fraction"] == pytest.approx(4 / leaf_mask.sum())


@pytest.mark.parametrize("groups", [("shape",), ("lesions",)])
def test_mask_dependent_features_require_masks(sample, groups):
    image, _, _ = sample
    with pytest.raises(ValueError):
        extract_features(image, groups=groups)


def test_lesions_require_explicit_lesion_mask(sample):
    image, leaf_mask, _ = sample
    with pytest.raises(ValueError):
        extract_features(image, leaf_mask, groups=("lesions",))


@pytest.mark.parametrize("groups", [("unknown",), ("color", "color"), ()])
def test_invalid_groups_are_rejected(sample, groups):
    image, _, _ = sample
    with pytest.raises(ValueError):
        extract_features(image, groups=groups)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "empty", "outside"])
def test_invalid_masks_are_rejected(sample, invalid):
    image, leaf_mask, lesion_mask = sample
    if invalid == "shape":
        leaf_mask = leaf_mask[:-1]
    elif invalid == "dtype":
        leaf_mask = leaf_mask.astype(np.uint8)
    elif invalid == "empty":
        leaf_mask[:] = False
    else:
        lesion_mask[0, 0] = True

    with pytest.raises(ValueError):
        extract_features(image, leaf_mask, lesion_mask, groups=FEATURE_GROUPS)


@pytest.mark.parametrize("invalid", ["grayscale", "rgba", "integer", "range", "nan"])
def test_invalid_images_are_rejected(sample, invalid):
    image, _, _ = sample
    if invalid == "grayscale":
        image = image[:, :, 0]
    elif invalid == "rgba":
        image = np.zeros((32, 32, 4), dtype=np.uint8)
    elif invalid == "integer":
        image = image.astype(np.int16)
    elif invalid == "range":
        image = image.astype(float)
    else:
        image = image.astype(float) / 255
        image[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        extract_features(image)


def test_float_rgb_matches_uint8_rgb(sample):
    image, leaf_mask, lesion_mask = sample
    integer = extract_features(image, leaf_mask, lesion_mask, groups=FEATURE_GROUPS)
    floating = extract_features(
        image.astype(float) / 255, leaf_mask, lesion_mask, groups=FEATURE_GROUPS
    )

    assert integer.keys() == floating.keys()
    np.testing.assert_allclose(
        list(integer.values()), list(floating.values()), atol=1e-5
    )


@pytest.mark.parametrize("group", ["glcm", "lbp"])
def test_insufficient_mask_neighborhood_is_rejected(sample, group):
    image, leaf_mask, _ = sample
    leaf_mask[:] = False
    leaf_mask[16, 16] = True

    with pytest.raises(ValueError):
        extract_features(image, leaf_mask, groups=(group,))


def test_raw_images_produce_only_available_feature_groups(sample):
    image, _, _ = sample
    result = extract_features(image)

    assert {name.split("_")[0] for name in result} == {"color", "glcm", "lbp"}
    assert np.isfinite(list(result.values())).all()


def test_extraction_preserves_inputs_and_column_order(sample):
    originals = [array.copy() for array in sample]

    first = extract_features(*sample, groups=FEATURE_GROUPS)
    second = extract_features(*sample, groups=FEATURE_GROUPS)

    for original, current in zip(originals, sample, strict=True):
        np.testing.assert_array_equal(current, original)
    assert list(first) == list(second)
    assert np.isfinite(list(first.values())).all()
    assert {name.split("_")[0] for name in first} == set(FEATURE_GROUPS)
