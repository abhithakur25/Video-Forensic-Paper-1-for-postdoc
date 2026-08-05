# -*- coding: utf-8 -*-
"""Honest evaluation metrics (sklearn). Do NOT use mealpy.metrics.confusion_matrix."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluation_metrics(y_true, y_pred):
    """Return [ACC, SEN, SPE, PRE, F1, BAL_ACC] as floats.

    SEN = recall of positive class (label 1) when binary; else macro recall.
    SPE = TN / (TN + FP) for binary.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch {len(y_true)} vs {len(y_pred)}")

    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    pre = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    sen = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        spe = float("nan")

    return [acc, sen, spe, pre, f1, bal]


def majority_baseline_accuracy(y):
    y = np.asarray(y).ravel()
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))
