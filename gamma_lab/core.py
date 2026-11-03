"""Leakage-aware splitting, operating-point selection and portable inference."""

import numpy as np
from sklearn.model_selection import train_test_split


def grouped_split(x, y, seed=42):
    """Stratify unique feature groups, then map them back to row indices."""
    _, first, groups = np.unique(x, axis=0, return_index=True, return_inverse=True)
    labels = y[first]
    if np.any(labels[groups] != y):
        raise ValueError("Identical feature vectors have conflicting labels")
    train, remaining = train_test_split(
        np.arange(len(first)), test_size=0.4, stratify=labels, random_state=seed
    )
    validation, test = train_test_split(
        remaining, test_size=0.5, stratify=labels[remaining], random_state=seed
    )
    return tuple(np.flatnonzero(np.isin(groups, part)) for part in (train, validation, test))


def choose_threshold(probabilities, labels, target=0.05):
    """Choose smallest observed score whose negative acceptance is within target."""
    negative = np.asarray(probabilities)[np.asarray(labels) == 0]
    if not len(negative) or not 0 <= target <= 1:
        raise ValueError("Need negatives and a target in [0, 1]")
    allowed = int(np.floor(target * len(negative)))
    if allowed == 0:
        return float(np.nextafter(np.max(negative), np.inf))
    ordered = np.sort(negative)[::-1]
    threshold = ordered[allowed - 1]
    if np.sum(negative >= threshold) > allowed:
        threshold = np.nextafter(threshold, np.inf)
    return float(threshold)


def predict(model, features):
    """Run a ReLU hidden network with a sigmoid output, using JSON weights."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]) or not np.isfinite(x).all():
        raise ValueError("Expected finite feature vectors of the model's input width")
    x = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    for i, (weights, biases) in enumerate(zip(model["weights"], model["biases"])):
        x = x @ np.asarray(weights) + np.asarray(biases)
        if i < len(model["weights"]) - 1:
            x = np.maximum(x, 0)
    return 1 / (1 + np.exp(-np.clip(x[:, 0], -700, 700)))
