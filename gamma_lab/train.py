"""Train one fixed neural classifier and write an auditable JSON artifact."""

import hashlib
import io
import json
import os
import resource
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from gamma_lab.core import choose_threshold, grouped_split, predict

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
SHA256 = "e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a"


def verify_bytes(raw):
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Dataset checksum mismatch")
    return raw


def export_model(scaler, model, threshold):
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": [w.tolist() for w in model.coefs_],
        "biases": [b.tolist() for b in model.intercepts_],
        "threshold": threshold,
    }


def main():
    start = time.monotonic()
    raw = verify_bytes(urllib.request.urlopen(URL, timeout=60).read())
    rows = np.genfromtxt(io.BytesIO(raw), delimiter=",", dtype=str)
    x = rows[:, :10].astype(float)
    y = (rows[:, 10] == "g").astype(int)
    train, validation, test = grouped_split(x, y)
    scaler = StandardScaler().fit(x[train])
    net = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        random_state=42,
        max_iter=200,
        early_stopping=False,
        alpha=0.001,
    ).fit(scaler.transform(x[train]), y[train])
    validation_scores = net.predict_proba(scaler.transform(x[validation]))[:, 1]
    threshold = choose_threshold(validation_scores, y[validation])
    model = export_model(scaler, net, threshold)
    scores = predict(model, x[test])
    assert np.allclose(scores, net.predict_proba(scaler.transform(x[test]))[:, 1])
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[test], predicted, labels=[0, 1]).ravel()
    baseline = DummyClassifier(strategy="most_frequent").fit(x[train], y[train])
    b_tn, b_fp, b_fn, b_tp = confusion_matrix(
        y[test], baseline.predict(x[test]), labels=[0, 1]
    ).ravel()
    folder = Path("model")
    folder.mkdir(exist_ok=True)
    model_bytes = (json.dumps(model, indent=2, allow_nan=False) + "\n").encode()
    (folder / "model.json").write_bytes(model_bytes)
    metrics = {
        "threshold": threshold,
        "validation_fpr": float(np.mean(validation_scores[y[validation] == 0] >= threshold)),
        "test": {
            "roc_auc": float(roc_auc_score(y[test], scores)),
            "fpr": float(fp / (fp + tn)),
            "recall": float(tp / (tp + fn)),
            "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        },
        "majority_baseline_confusion_tn_fp_fn_tp": [int(b_tn), int(b_fp), int(b_fn), int(b_tp)],
        "split_counts": {
            name: {"total": len(idx), "gamma": int(y[idx].sum())}
            for name, idx in zip(("train", "validation", "test"), (train, validation, test))
        },
        "duration_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "iterations": net.n_iter_,
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "training_type": "from-scratch",
        "dataset_license": "CC-BY-4.0",
        "dataset_url": URL,
        "dataset_sha256": SHA256,
        "training_commit": commit,
        "split_method": "seed 42, stratified unique feature groups 60/20/20",
        "artifact_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "seed": 42,
        "python": os.sys.version,
        "model": "sklearn MLPClassifier (32,16)",
    }
    for name, value in (("metrics.json", metrics), ("training-manifest.json", manifest)):
        (folder / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
