import numpy as np
import pytest

from gamma_lab.core import choose_threshold, grouped_split, predict


def test_grouped_split_is_deterministic_and_disjoint():
    x = np.repeat(np.arange(100)[:, None], 2, axis=0)
    y = np.repeat(np.arange(100) % 2, 2)
    parts = grouped_split(x, y)
    assert all(np.array_equal(a, b) for a, b in zip(parts, grouped_split(x, y)))
    groups = [set(x[index, 0]) for index in parts]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    assert sum(map(len, parts)) == len(x)


def test_threshold_limits_validation_false_positives():
    probabilities = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    labels = np.array([0, 0, 0, 0, 1])
    threshold = choose_threshold(probabilities, labels, 0.25)
    assert np.mean(probabilities[labels == 0] >= threshold) <= 0.25
    assert threshold == 0.9


def test_portable_predict_known_network():
    model = {"mean": [1.0], "scale": [2.0], "weights": [[[2.0]]], "biases": [[0.0]]}
    assert predict(model, [[3.0]])[0] == pytest.approx(1 / (1 + np.exp(-2)))


def test_rejects_conflicting_duplicate_labels():
    with pytest.raises(ValueError, match="conflicting"):
        grouped_split(np.array([[1], [1]]), np.array([0, 1]))


def test_model_export_matches_sklearn():
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    from gamma_lab.train import export_model

    x = np.arange(40, dtype=float).reshape(20, 2)
    y = np.tile([0, 1], 10)
    scaler = StandardScaler().fit(x)
    net = MLPClassifier(hidden_layer_sizes=(4,), random_state=42, max_iter=20).fit(
        scaler.transform(x), y
    )
    model = export_model(scaler, net, 0.5)
    assert np.allclose(predict(model, x), net.predict_proba(scaler.transform(x))[:, 1])


def test_download_rejects_unexpected_checksum():
    from gamma_lab.train import verify_bytes

    with pytest.raises(ValueError, match="checksum"):
        verify_bytes(b"not the dataset")
