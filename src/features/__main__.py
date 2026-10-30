"""Export named feature columns separately from sample IDs and labels."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from .dataset import build_feature_matrix, load_folder
from .extract import (
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    GLCM_ANGLES,
    GLCM_DISTANCES,
    GLCM_LEVELS,
    GLCM_PROPERTIES,
    LBP_SETTINGS,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder containing class/image files")
    parser.add_argument(
        "output", type=Path, help="Destination for X.csv, samples.csv and metadata.json"
    )
    parser.add_argument("--masks", type=Path, help="Leaf masks: class/image.png")
    parser.add_argument("--lesions", type=Path, help="Lesion masks: class/image.png")
    parser.add_argument(
        "--groups", nargs="+", choices=FEATURE_GROUPS, default=DEFAULT_GROUPS
    )
    args = parser.parse_args()
    if args.lesions is not None and args.masks is None:
        parser.error("--lesions requires --masks")
    if "shape" in args.groups and args.masks is None:
        parser.error("shape requires --masks")
    if "lesions" in args.groups and (args.masks is None or args.lesions is None):
        parser.error("lesions requires --masks and --lesions")

    X, samples = build_feature_matrix(
        load_folder(args.input, masks=args.masks, lesions=args.lesions),
        groups=args.groups,
    )
    metadata = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "input_scope": "whole_image" if args.masks is None else "leaf_mask",
        "leaf_masks": str(args.masks.resolve()) if args.masks is not None else None,
        "lesion_masks": str(args.lesions.resolve())
        if args.lesions is not None
        else None,
        "groups": [g for g in FEATURE_GROUPS if g in args.groups],
        "columns": list(X.columns),
        "rows": len(X),
        "class_counts": {
            str(k): int(v) for k, v in samples.label.value_counts().items()
        },
        "parameters": {
            "glcm_levels": GLCM_LEVELS,
            "glcm_distances": GLCM_DISTANCES,
            "glcm_angles_radians": GLCM_ANGLES,
            "glcm_properties": GLCM_PROPERTIES,
            "lbp_radius_points": LBP_SETTINGS,
            "lbp_method": "uniform",
            "color": "Lab: mean and population standard deviation per channel",
            "resizing": None,
        },
        "versions": {
            name: version(name)
            for name in ("numpy", "pandas", "pillow", "scikit-image")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    X.to_csv(args.output / "X.csv", index=False)
    samples.to_csv(args.output / "samples.csv", index=False)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {X.shape[0]} samples x {X.shape[1]} features to {args.output}")
    if args.masks is None:
        print(
            "No leaf masks: features include background; comparisons are provisional."
        )


if __name__ == "__main__":
    main()
