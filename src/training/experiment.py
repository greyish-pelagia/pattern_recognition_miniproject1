"""Group-aware split, grid search over (variant x feature set x model), reporting."""

import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import BASELINE, VARIANTS, load_and_preprocess, split_features
from .plots import show_images

SEED = 42

# model name -> (classifier, parameter grid); each classifier follows a StandardScaler
MODELS = {
    "SVM (RBF)": (
        SVC(kernel="rbf"),
        {"clf__C": [0.1, 1, 10, 100, 1000], "clf__gamma": ["scale", 1e-4, 1e-3, 1e-2, 0.1]},
    ),
    "SVM (linear)": (SVC(kernel="linear"), {"clf__C": [0.01, 0.1, 1, 10]}),
    "KNN": (
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [1, 3, 5, 7, 11, 15, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 10],
            "clf__min_samples_leaf": [1, 3],
        },
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=5000),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
}


@dataclass(frozen=True)
class Split:
    """Row indices of the train and test parts, plus labels and duplicate-image groups."""

    y: np.ndarray
    groups: np.ndarray
    train: np.ndarray
    test: np.ndarray

    @property
    def y_train(self):
        return self.y[self.train]

    @property
    def y_test(self):
        return self.y[self.test]

    @property
    def groups_train(self):
        return self.groups[self.train]


def group_train_test_split(y, groups, test_size=0.3, seed=SEED):
    """Stratified train/test indices where identical images (groups) stay together.

    sklearn has no stratified group shuffle split, so the test set is the union
    of the first k folds of a 10-fold StratifiedGroupKFold (k/10 ~= test_size).
    """
    n_folds = 10
    folds = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
    test_folds = list(folds.split(np.zeros(len(y)), y, groups))[: round(test_size * n_folds)]
    test = np.sort(np.concatenate([fold_test for _, fold_test in test_folds]))
    return np.setdiff1d(np.arange(len(y)), test), test


def make_split(frames, test_size=0.3, seed=SEED):
    """Split the rows shared by all frames; groups are hashes of the pixels."""
    y = frames[BASELINE]["label"].to_numpy()
    groups = pd.factorize(frames[BASELINE]["image_sha256"])[0]
    train, test = group_train_test_split(y, groups, test_size, seed)
    return Split(y, groups, train, test)


def make_pipeline(estimator):
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def _cv(n_splits):
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)


def run_search(frames, split, n_splits=5, n_jobs=-1):
    """GridSearchCV on the train part for every (variant, model).

    Returns (results table sorted by CV f1_macro, fitted searches). Every search
    uses the same folds. The sort is stable, so on a tie the configuration that
    was searched first wins (the baseline variant, then the order of MODELS).
    """
    rows, searches = [], {}
    for variant, df in frames.items():
        X, _ = split_features(df)
        for model, (estimator, grid) in MODELS.items():
            search = GridSearchCV(
                make_pipeline(estimator), grid, cv=_cv(n_splits),
                scoring="f1_macro", n_jobs=n_jobs,
            )
            search.fit(X.iloc[split.train], split.y_train, groups=split.groups_train)
            rows.append(
                {
                    "variant": variant,
                    "model": model,
                    "cv_f1_macro": search.best_score_,
                    "cv_std": search.cv_results_["std_test_score"][search.best_index_],
                    "best_params": {
                        k.removeprefix("clf__"): v for k, v in search.best_params_.items()
                    },
                }
            )
            searches[(variant, model)] = search
    results = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False, kind="stable")
    return results.reset_index(drop=True), searches


def best_per_model(results):
    """Best configuration per model by CV f1_macro, best model first."""
    best = results.loc[results.groupby("model")["cv_f1_macro"].idxmax()]
    return best.sort_values("cv_f1_macro", ascending=False, kind="stable").reset_index(drop=True)


def _test_features(row, searches, frames, split):
    X, _ = split_features(frames[row.variant])
    return searches[(row.variant, row.model)], X.iloc[split.test]


def evaluate_on_test(best, searches, frames, split):
    """Score each model's CV-selected configuration on the test set.

    Returns (metrics table, {model: predicted labels}). Precision, recall and F1
    treat label 1 as the positive class; f1_macro averages both classes.
    """
    y = split.y_test
    rows, predictions = [], {}
    for row in best.itertuples():
        search, X_test = _test_features(row, searches, frames, split)
        pred = predictions[row.model] = search.predict(X_test)
        rows.append(
            {
                "model": row.model,
                "variant": row.variant,
                "cv_f1_macro": row.cv_f1_macro,
                "test_accuracy": accuracy_score(y, pred),
                "test_precision": precision_score(y, pred, zero_division=0),
                "test_recall": recall_score(y, pred),
                "test_f1": f1_score(y, pred),
                "test_f1_macro": f1_score(y, pred, average="macro"),
            }
        )
    return pd.DataFrame(rows), predictions


def error_overlap(predictions, y_test):
    """Number of test images misclassified by at least one, two and all models."""
    n_wrong = np.sum([pred != y_test for pred in predictions.values()], axis=0)
    return pd.Series(
        {
            "at least one model": (n_wrong >= 1).sum(),
            "two or more models": (n_wrong >= 2).sum(),
            f"all {len(predictions)} models": (n_wrong == len(predictions)).sum(),
        }
    )


def feature_importance(winner, searches, frames, split, n_repeats=20):
    """Drop in test f1_macro when features are shuffled, per group and per feature.

    A group is shuffled as a whole (same row permutation for all its columns).
    Returns two Series, each sorted from the least to the most important.
    """
    search, X_test = _test_features(winner, searches, frames, split)
    rng = np.random.default_rng(SEED)

    def score(X):
        return f1_score(split.y_test, search.predict(X), average="macro")

    baseline = score(X_test)

    def drop(columns):
        drops = []
        for _ in range(n_repeats):
            X = X_test.copy()
            X[columns] = X_test[columns].to_numpy()[rng.permutation(len(X))]
            drops.append(baseline - score(X))
        return np.mean(drops)

    by_group = {}
    for column in X_test.columns:
        by_group.setdefault(column.split("_")[0], []).append(column)
    return (
        pd.Series({group: drop(columns) for group, columns in by_group.items()}).sort_values(),
        pd.Series({column: drop([column]) for column in X_test.columns}).sort_values(),
    )


def variant_comparison(results):
    """Best CV f1_macro per (model, variant)."""
    return results.pivot_table(
        index="model", columns="variant", values="cv_f1_macro", aggfunc="max"
    ).round(4)


def plot_errors(best, predictions, frames, split, data_dir, label_to_name, max_images=8):
    """Show the test images that most models get wrong.

    Each image appears as the original and as preprocessed with the best model's variant.
    """
    wrong = pd.DataFrame({model: pred != split.y_test for model, pred in predictions.items()})
    n_wrong = wrong.sum(axis=1).sort_values(ascending=False, kind="stable")
    shown = n_wrong[n_wrong > 0].index[:max_images]
    if shown.empty:
        print("No test mistakes.")
        return
    variant = best.iloc[0].variant
    sample_ids = frames[BASELINE]["sample_id"].to_numpy()[split.test]
    images, titles = [], []
    for i in shown:
        original, processed, _ = load_and_preprocess(
            Path(data_dir) / sample_ids[i], VARIANTS[variant]
        )
        missed_by = textwrap.fill(", ".join(wrong.columns[wrong.loc[i].to_numpy()]), 34)
        images += [original, processed]
        titles += [f"true: {label_to_name[split.y_test[i]]}\nmissed by: {missed_by}", variant]
    n_missed = int((n_wrong > 0).sum())
    title = f"{len(shown)} of {n_missed} misclassified test images, most missed first"
    show_images(images, titles, ncols=4, title=title)
