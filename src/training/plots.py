"""Plots shared by the training notebooks."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


def show_images(images, titles, ncols, title=None):
    """Show images in a grid, `ncols` per row."""
    nrows = -(-len(images) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.8 * nrows), squeeze=False, layout="constrained"
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, image_title in zip(axes.flat, images, titles, strict=False):
        ax.imshow(image)
        ax.set_title(image_title, fontsize=9)
    fig.suptitle(title)
    plt.show()


def show_samples(paths, labels, label_to_name, per_class=5, seed=42):
    """Show random images, one row per class."""
    rng = np.random.default_rng(seed)
    images, titles = [], []
    for label, name in label_to_name.items():
        for i in rng.choice(np.flatnonzero(labels == label), per_class, replace=False):
            with Image.open(paths[i]) as image:
                images.append(image.convert("RGB"))
            titles.append(f"{name} (#{i})")
    show_images(images, titles, ncols=per_class)


def plot_confusion_matrices(predictions, y_true, label_to_name, ncols=3):
    """One confusion matrix per model; `predictions` maps model name -> predicted labels."""
    nrows = -(-len(predictions) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    for ax, (model, y_pred) in zip(axes.flat, predictions.items(), strict=False):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=list(label_to_name.values()),
            ax=ax, colorbar=False, cmap="Blues",
        )
        ax.set_title(model)
    for ax in list(axes.flat)[len(predictions):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def plot_feature_importance(by_group, by_feature, top=10):
    """Bar charts of `feature_importance` output: all groups and the `top` features."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    by_group.plot.barh(ax=left, title="Feature groups")
    by_feature.tail(top).plot.barh(ax=right, title=f"Top {top} features")
    for ax in (left, right):
        ax.set_xlabel("Drop in test f1_macro when shuffled")
    plt.show()


def plot_feature_space(X, y, label_to_name, top_features):
    """Scatter of the two most important features next to a PCA of all features."""
    pca = PCA(2).fit_transform(StandardScaler().fit_transform(X))
    panels = [
        (X[list(top_features)].to_numpy(), top_features, "Two most important features"),
        (pca, ["PC1", "PC2"], "PCA of all features"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, (xy, (xlabel, ylabel), title) in zip(axes, panels, strict=True):
        for label, name in label_to_name.items():
            ax.scatter(*xy[y == label].T, s=12, alpha=0.5, label=name)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axes[0].legend()
    plt.show()
