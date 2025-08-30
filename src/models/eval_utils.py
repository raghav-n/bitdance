from typing import Dict, List
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, label_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute per-label precision/recall/F1 and average precision (PR-AUC).

    Args:
        y_true: shape (N, L) binary array
        y_prob: shape (N, L) float array in [0,1]
        label_names: list of label names length L
    Returns: dict mapping label -> metrics
    """
    assert y_true.shape == y_prob.shape, "Shapes of y_true and y_prob must match"
    y_pred = (y_prob >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    ap = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])]
    return {
        label_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "ap": float(ap[i]),
        }
        for i in range(len(label_names))
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, label_names: List[str], grid=None) -> Dict[str, float]:
    """Find best threshold per label by F1 over a grid.
    Returns dict label->best_threshold.
    """
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)
    best = {}
    for i, name in enumerate(label_names):
        y = y_true[:, i]
        p = y_prob[:, i]
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            pred = (p >= t).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best[name] = best_t
    return best

