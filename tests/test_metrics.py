import numpy as np
from malaria_vit.metrics import compute_metrics

def test_perfect_predictions():
    labels = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_metrics(labels, preds, probs)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0

def test_keys_present():
    labels = np.array([0, 1, 1, 0]); preds = np.array([0, 1, 0, 0])
    probs = np.array([0.2, 0.7, 0.4, 0.3])
    m = compute_metrics(labels, preds, probs)
    assert set(m) >= {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}
