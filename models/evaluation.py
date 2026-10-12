"""Numpy-based evaluation helpers for dropout prediction models."""

from __future__ import annotations

import numpy as np


def _classes(y_true, y_pred):
    return np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute multiclass metrics without sklearn dependency."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = _classes(y_true, y_pred)

    per_class = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        balanced = (recall + specificity) / 2
        per_class.append((precision, recall, f1, balanced))

    arr = np.asarray(per_class, dtype=float)
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        # Standard multiclass balanced accuracy is macro-averaged recall.
        "balanced_accuracy": float(arr[:, 1].mean()),
        "macro_f1": float(arr[:, 2].mean()),
        "macro_precision": float(arr[:, 0].mean()),
        "macro_recall": float(arr[:, 1].mean()),
    }

    if y_proba is not None:
        metrics["log_loss"] = multiclass_log_loss(y_true, y_proba)
    return metrics


def multiclass_log_loss(y_true, y_proba, labels=None, eps=1e-15) -> float:
    """Compute multiclass log-loss."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    if labels is None:
        labels = np.unique(y_true)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    indices = np.asarray([label_to_idx[label] for label in y_true])
    clipped = np.clip(y_proba, eps, 1 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(clipped[np.arange(len(indices)), indices])))


def binary_probability_metrics(y_true_binary, y_score) -> dict:
    """Compute ROC-AUC and PR-AUC with numpy only."""
    y_true = np.asarray(y_true_binary).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    return {
        "roc_auc": roc_auc_score_numpy(y_true, y_score),
        "pr_auc": average_precision_numpy(y_true, y_score),
    }


def roc_auc_score_numpy(y_true, y_score) -> float:
    """Compute ROC-AUC using rank statistics."""
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = np.sum(pos)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_numpy(y_true, y_score) -> float:
    """Compute average precision for binary labels."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    positives = np.sum(y_sorted == 1)
    if positives == 0:
        return float("nan")
    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float(np.sum(precision[y_sorted == 1]) / positives)
