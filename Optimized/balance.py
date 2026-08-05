# -*- coding: utf-8 -*-
"""Class weights and minority oversampling for Paper 1 training."""
from __future__ import annotations

import numpy as np


def compute_class_weight_dict(y, minority_scale=1.0):
    """sklearn-style balanced weights, with extra scale on minority class.

    w_c = n / (n_classes * count_c)
    then multiply minority class weight by minority_scale.
    """
    y = np.asarray(y).ravel().astype(int)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_cls = len(classes)
    w = {}
    for c, cnt in zip(classes, counts):
        w[int(c)] = float(n / (n_cls * max(cnt, 1)))
    if minority_scale != 1.0 and len(classes) >= 2:
        min_c = int(classes[np.argmin(counts)])
        w[min_c] = float(w[min_c] * float(minority_scale))
    return w


def keras_class_weight(y, minority_scale=1.0):
    """Dict for model.fit(class_weight=...)."""
    return compute_class_weight_dict(y, minority_scale=minority_scale)


def oversample_minority(x, y, seed=0, ratio=1.0):
    """Random oversample minority toward majority * ratio (train only).

    ratio=1.0 -> equal counts; ratio=1.5 -> minority oversampled to 1.5x majority.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y).ravel().astype(int)
    x = np.asarray(x)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return x, y
    maj = int(counts.max())
    target = max(maj, int(round(maj * float(ratio))))
    parts_x, parts_y = [], []
    for c, cnt in zip(classes, counts):
        idx = np.where(y == c)[0]
        if cnt < target:
            extra = rng.choice(idx, size=target - cnt, replace=True)
            idx = np.concatenate([idx, extra])
        rng.shuffle(idx)
        parts_x.append(x[idx])
        parts_y.append(y[idx])
    x_out = np.concatenate(parts_x, axis=0)
    y_out = np.concatenate(parts_y, axis=0)
    perm = rng.permutation(len(y_out))
    return x_out[perm], y_out[perm]
