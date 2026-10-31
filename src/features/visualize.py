"""Visualize descriptors, honoring optional sample splits supplied by training."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm
from PIL import Image
from skimage import color, feature, util
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .extract import GLCM_ANGLES, GLCM_LEVELS, extract_features

CLASS_STYLE = {
    "Potato___healthy": ("Zdrowy", "#2260A8", "o"),
    "Potato___Early_blight": ("Early blight", "#C26A16", "^"),
    "Potato___Late_blight": ("Late blight", "#8352A0", "s"),
}
GROUP_LABELS = {"color": "Barwa", "glcm": "GLCM", "lbp": "LBP"}
LBP_COLORS = (
    "#234F80",
    "#A8D0E6",
    "#D49B22",
    "#8D5C22",
    "#899C3D",
    "#D2DBA2",
    "#AD5C83",
    "#E4B8CF",
    "#6B5D99",
    "#CECBD8",
)
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "text.color": "#252B32",
    "axes.labelcolor": "#252B32",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.18,
    "savefig.facecolor": "white",
}
DPI = 170


def _read_dataset(folder):
    folder = Path(folder)
    X = pd.read_csv(folder / "X.csv")
    samples = pd.read_csv(folder / "samples.csv", dtype=str, keep_default_na=False)
    metadata = json.loads((folder / "metadata.json").read_text())
    if len(X) != len(samples):
        raise ValueError("X and samples must have aligned rows")
    if samples.sample_id.duplicated().any() or samples.sample_id.eq("").any():
        raise ValueError("Sample identifiers must be nonempty and unique")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features must be finite")
    if metadata["input_scope"] != "whole_image":
        raise ValueError("These raw-image previews require input_scope=whole_image")

    # Training owns splits. Without one, explicitly visualize every supplied row.
    selected = pd.Series(True, index=samples.index)
    scope = "all_samples"
    scope_label = "Wszystkie przykłady (brak split.csv)"
    split_path = folder / "split.csv"
    if split_path.exists():
        splits = pd.read_csv(
            split_path, dtype={"sample_id": str}, keep_default_na=False
        )
        if not samples.sample_id.equals(splits.sample_id):
            raise ValueError("Samples and splits must have aligned sample identifiers")
        if not set(splits["split"]).issubset({"development", "test"}):
            raise ValueError("Unknown split value")
        selected = splits["split"].eq("development")
        scope = "development"
        scope_label = "Development"
    if not selected.any():
        raise ValueError("No samples selected for visualization")
    hashes = set(samples.image_sha256)
    samples = samples.loc[selected].reset_index(drop=True)
    if set(samples.label) - set(CLASS_STYLE):
        raise ValueError("Unknown potato class label")
    return {
        "folder": folder,
        "name": "Easy" if "Potato___healthy" in set(samples.label) else "Hard",
        "X": X.loc[selected].reset_index(drop=True),
        "samples": samples,
        "metadata": metadata,
        "source_hashes": hashes,
        "scope": scope,
        "scope_label": scope_label,
    }


def _plot_feature_spaces(datasets, output):
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(15, 5 * len(datasets)), squeeze=False
    )
    projections = []
    for row, data in enumerate(datasets):
        X, samples = data["X"], data["samples"]
        for column, group in enumerate(("color", "glcm", "lbp")):
            ax = axes[row, column]
            if group == "color":
                coordinates = X[["color_a_mean", "color_b_mean"]].to_numpy()
                ax.set_xlabel("Średnie a* (zieleń → czerwień)")
                ax.set_ylabel("Średnie b* (błękit → żółć)")
                title = "Barwa: 2 z 6 rzeczywistych cech"
            else:
                columns = [c for c in X.columns if c.startswith(f"{group}_")]
                model = PCA(n_components=2)
                coordinates = model.fit_transform(
                    StandardScaler().fit_transform(X[columns])
                )
                variance = model.explained_variance_ratio_
                ax.set_xlabel(f"PC1 ({variance[0]:.1%} wariancji)")
                ax.set_ylabel(f"PC2 ({variance[1]:.1%} wariancji)")
                title = f"{GROUP_LABELS[group]}: rzut PCA {len(columns)} cech"
                projections.append(
                    {
                        "dataset": data["name"],
                        "group": group,
                        "explained_variance_ratio": variance.tolist(),
                    }
                )
            for label in sorted(samples.label.unique()):
                name, shade, marker = CLASS_STYLE[label]
                selected = samples.label.eq(label)
                ax.scatter(
                    *coordinates[selected].T,
                    s=17,
                    alpha=0.45,
                    color=shade,
                    marker=marker,
                    linewidths=0,
                    label=f"{name} (n={selected.sum()})",
                )
            ax.set_title(f"{data['name']} · {title}", loc="left")
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Co rozdziela klasy w przestrzeni cech?", x=0.07, ha="left", fontsize=18
    )
    fig.text(
        0.07,
        0.015,
        "; ".join(f"{data['name']}: {data['scope_label']}" for data in datasets)
        + ". Cały kadr z tłem.\n"
        "PCA jest wizualizacją, nie klasyfikatorem. Nakładanie się punktów w 2D "
        "nie rozstrzyga o separacji w pełnej przestrzeni.",
        fontsize=9,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    fig.savefig(output / "feature_spaces.png", dpi=DPI)
    plt.close(fig)
    return projections


def _plot_examples(data, output):
    X, samples = data["X"], data["samples"]
    columns = [c for c in X.columns if c.startswith("color_")]
    standardized = StandardScaler().fit_transform(X[columns])
    examples = []
    for label in sorted(samples.label.unique()):
        indices = np.flatnonzero(samples.label.eq(label))
        values = standardized[indices]
        position = indices[
            np.argmin(np.square(values - np.median(values, axis=0)).sum(axis=1))
        ]
        sample_id = samples.iloc[position].sample_id
        with Image.open(Path(data["metadata"]["input"]) / sample_id) as source:
            rgb = np.array(source.convert("RGB"))
        image = util.img_as_float32(rgb)
        lab = color.rgb2lab(image)
        gray = util.img_as_ubyte(color.rgb2gray(image))
        quantized = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
        glcm = feature.graycomatrix(
            quantized, [1], GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True
        ).mean(axis=(2, 3))
        lbp = feature.local_binary_pattern(gray, 8, 1, method="uniform")
        interior = np.zeros(gray.shape, dtype=bool)
        interior[1:-1, 1:-1] = True
        lbp = np.ma.array(lbp, mask=~interior)
        features = extract_features(rgb, groups=("lbp",))
        histogram = [features[f"lbp_r1_{i}"] for i in range(10)]
        examples.append(
            {
                "sample_id": sample_id,
                "label": label,
                "rgb": rgb,
                "lab": lab,
                "glcm": glcm,
                "lbp": lbp,
                "histogram": histogram,
            }
        )

    fig, axes = plt.subplots(
        len(examples), 6, figsize=(18, 3.9 * len(examples)), squeeze=False
    )
    limits = {
        channel: (
            min(e["lab"][:, :, channel].min() for e in examples),
            max(e["lab"][:, :, channel].max() for e in examples),
        )
        for channel in (1, 2)
    }
    histogram_limit = max(max(example["histogram"]) for example in examples) * 1.2
    for row, example in enumerate(examples):
        axes[row, 0].imshow(example["rgb"])
        axes[row, 0].set_title(f"{CLASS_STYLE[example['label']][0]} · RGB")
        for column, channel in ((1, 1), (2, 2)):
            low, high = limits[channel]
            image = axes[row, column].imshow(
                example["lab"][:, :, channel], cmap="coolwarm", vmin=low, vmax=high
            )
            axes[row, column].set_title("Lab a*" if channel == 1 else "Lab b*")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
        image = axes[row, 3].imshow(
            example["glcm"],
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=1e-5, vmax=1, clip=True),
        )
        axes[row, 3].set_title("GLCM · d=1 · skala log")
        axes[row, 3].set_xlabel("Poziom j (0–31)")
        axes[row, 3].set_ylabel("Poziom i (0–31)")
        bar = fig.colorbar(image, ax=axes[row, 3], fraction=0.046, pad=0.03)
        bar.set_ticks(
            [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            labels=["≤10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "1"],
        )
        axes[row, 4].imshow(
            example["lbp"], cmap=ListedColormap(LBP_COLORS), vmin=-0.5, vmax=9.5
        )
        axes[row, 4].set_title("Kody LBP · r=1")
        axes[row, 5].bar(np.arange(10), example["histogram"], color=LBP_COLORS)
        axes[row, 5].set(
            title="Histogram LBP",
            xlabel="Kod wzorca",
            ylabel="Udział sąsiedztw",
            ylim=(0, histogram_limit),
        )
        axes[row, 5].set_xticks(np.arange(10))
        for column in (0, 1, 2, 4):
            axes[row, column].axis("off")
        axes[row, 3].grid(False)
    fig.suptitle(
        f"{data['name']} · Jak obraz zamienia się w cechy",
        x=0.035,
        ha="left",
        fontsize=18,
    )
    fig.text(
        0.035,
        0.02,
        f"{data['scope_label']}: przykład najbliższy medianie 6 standaryzowanych cech barwy w swojej klasie. "
        "Wspólne skale kanałów w obrębie planszy. Cały kadr z tłem.\n"
        "Mapy pokazują przekształcenia obrazu, nie lokalizację choroby. "
        "Barwa obejmuje też L*; pełne LBP zawiera również promień 2.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    filename = f"example_maps_{data['name'].lower()}.png"
    fig.savefig(output / filename, dpi=DPI)
    plt.close(fig)
    return [{"sample_id": e["sample_id"], "label": e["label"]} for e in examples]


def render_visualizations(folders, output):
    """Render descriptor figures; use development rows when a split is supplied."""
    datasets = [_read_dataset(folder) for folder in folders]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        projections = _plot_feature_spaces(datasets, output)
        representatives = {
            data["name"]: _plot_examples(data, output) for data in datasets
        }
    manifest = {
        "scope": {data["name"]: data["scope"] for data in datasets},
        "split_policy": "Use development rows from optional split.csv; otherwise plot all supplied samples.",
        "overlap_scope": "full input image hashes",
        "input_scope": "whole_image; includes background",
        "dpi": DPI,
        "representative_selection": "nearest within-class median of standardized six color features",
        "representatives": representatives,
        "pca": projections,
        "plotted_samples": {
            data["name"]: data["samples"].sample_id.tolist() for data in datasets
        },
        "sources": [
            {
                "export_folder": str(data["folder"].resolve()),
                "image_root": data["metadata"]["input"],
                "scope": data["scope"],
                "sample_count": len(data["samples"]),
            }
            for data in datasets
        ],
        "source_overlap": [
            {
                "datasets": [first["name"], second["name"]],
                "shared_image_count": len(
                    first["source_hashes"] & second["source_hashes"]
                ),
            }
            for first, second in combinations(datasets, 2)
        ],
        "limitations": "Color scatter displays only two of six features; PCA is an exploratory 2D projection. "
        "Image maps are not disease localization. Raw-image previews require whole_image inputs.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folders",
        type=Path,
        nargs="+",
        help="Feature export folders; optional split.csv selects development samples",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/visualizations"))
    args = parser.parse_args()
    render_visualizations(args.folders, args.output)
    print(f"Saved feature visualizations to {args.output}")


if __name__ == "__main__":
    main()
