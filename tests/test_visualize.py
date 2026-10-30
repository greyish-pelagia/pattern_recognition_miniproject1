import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from src.features.visualize import _read_dataset


@pytest.fixture
def exported_folder(tmp_path):
    pd.DataFrame(
        {"color_a_mean": [1.0, 2.0, 1e9], "color_b_mean": [3.0, 4.0, -1e9]}
    ).to_csv(tmp_path / "X.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": ["0001", "0002", "0003"],
            "label": ["Potato___healthy", "Potato___Early_blight", "Potato___healthy"],
            "image_sha256": ["hash-1", "hash-2", "hash-3"],
        }
    ).to_csv(tmp_path / "samples.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": ["0001", "0002", "0003"],
            "split": ["development", "development", "test"],
        }
    ).to_csv(tmp_path / "split.csv", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"input_scope": "whole_image"}))
    return tmp_path


def test_visualization_input_excludes_reserved_test_and_preserves_ids(exported_folder):
    data = _read_dataset(exported_folder)

    assert data["samples"].sample_id.tolist() == ["0001", "0002"]
    np.testing.assert_array_equal(data["X"].to_numpy(), [[1.0, 3.0], [2.0, 4.0]])
    assert data["X"].index.equals(data["samples"].index)
    assert data["scope"] == "development"


def test_visualization_rejects_misaligned_sample_identifiers(exported_folder):
    split = pd.read_csv(exported_folder / "split.csv", dtype=str)
    split.iloc[::-1].to_csv(exported_folder / "split.csv", index=False)

    with pytest.raises(ValueError, match="aligned"):
        _read_dataset(exported_folder)


def test_raw_previews_do_not_ignore_preprocessing_masks(exported_folder):
    (exported_folder / "metadata.json").write_text(
        json.dumps({"input_scope": "leaf_mask"})
    )

    with pytest.raises(ValueError, match="whole_image"):
        _read_dataset(exported_folder)


def test_visualization_without_split_includes_all_supplied_rows(exported_folder):
    (exported_folder / "split.csv").unlink()

    data = _read_dataset(exported_folder)

    assert data["samples"].sample_id.tolist() == ["0001", "0002", "0003"]
    assert data["X"].iloc[2].tolist() == [1e9, -1e9]
    assert data["scope"] == "all_samples"
    assert "brak split.csv" in data["scope_label"]
