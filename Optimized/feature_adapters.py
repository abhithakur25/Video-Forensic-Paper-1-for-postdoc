# -*- coding: utf-8 -*-
"""Adapters for Paper 1 Features.pkl tensors.

Shipped layout (N=50):
  comparative1/2/3/5 : (N, 128, 128, 10)
  comparative4       : (N, 10, 12)
  proposed           : (N, 10, 128, 128, 12)   # T, H, W, C
  labels             : (N,)
"""
from __future__ import annotations

import numpy as np


def as_spatial_from_proposed(x: np.ndarray, size: int = 32) -> np.ndarray:
    """(N,T,H,W,C) -> time-mean (N,size,size,C) via resize."""
    import cv2

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 5:
        raise ValueError(f"expected rank-5 proposed, got {x.shape}")
    vol = np.mean(x, axis=1)  # (N,H,W,C)
    n, h, w, c = vol.shape
    if h == size and w == size:
        return vol
    out = np.zeros((n, size, size, c), dtype=np.float32)
    for i in range(n):
        # cv2.resize works on last-channel images; for C>3 loop channels
        planes = []
        for ch in range(c):
            planes.append(
                cv2.resize(vol[i, :, :, ch], (size, size), interpolation=cv2.INTER_AREA)
            )
        out[i] = np.stack(planes, axis=-1)
    return out


def as_volume(x: np.ndarray, size: int = 32) -> np.ndarray:
    """(N,T,H,W,C) -> (N,T,size,size,C) for 3D-CNN / LSTM path."""
    import cv2

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 5:
        raise ValueError(f"expected rank-5, got {x.shape}")
    n, t, h, w, c = x.shape
    if h == size and w == size:
        return x
    out = np.zeros((n, t, size, size, c), dtype=np.float32)
    for i in range(n):
        for j in range(t):
            planes = [
                cv2.resize(x[i, j, :, :, ch], (size, size), interpolation=cv2.INTER_AREA)
                for ch in range(c)
            ]
            out[i, j] = np.stack(planes, axis=-1)
    return out


def as_spatial_2d(x: np.ndarray, size: int = 32) -> np.ndarray:
    """(N,H,W,C) -> (N,size,size,C)."""
    import cv2

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"expected rank-4, got {x.shape}")
    n, h, w, c = x.shape
    if h == size and w == size:
        return x
    out = np.zeros((n, size, size, c), dtype=np.float32)
    for i in range(n):
        planes = [
            cv2.resize(x[i, :, :, ch], (size, size), interpolation=cv2.INTER_AREA)
            for ch in range(c)
        ]
        out[i] = np.stack(planes, axis=-1)
    return out


def as_rgb_image(x: np.ndarray, size: int = 64) -> np.ndarray:
    """Project spatial or proposed features to RGB-like (N,size,size,3) in [0,255]."""
    import cv2

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 5:
        vol = as_spatial_from_proposed(x, size=size)
    elif x.ndim == 4:
        vol = as_spatial_2d(x, size=size)
    else:
        raise ValueError(f"as_rgb_image expects rank 4/5, got {x.shape}")

    c = vol.shape[-1]
    idx = [0, min(3, c - 1), min(6, c - 1)]
    ch = vol[:, :, :, idx]
    out = np.zeros((ch.shape[0], size, size, 3), dtype=np.float32)
    for i in range(ch.shape[0]):
        plane = ch[i]
        lo = plane.min(axis=(0, 1), keepdims=True)
        hi = plane.max(axis=(0, 1), keepdims=True)
        denom = np.maximum(hi - lo, 1e-6)
        normed = (plane - lo) / denom
        img = (normed * 255.0).astype(np.uint8)
        if img.shape[0] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        out[i] = img.astype(np.float32)
    return out


def as_flat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1)


def train_test_split_arrays(features, labels, train_size=0.8, seed=42):
    """Stratified-ish split by shuffle with fixed seed (small N)."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels).ravel()
    n = len(labels)
    idx = np.arange(n)
    # shuffle within each class for balance
    parts = []
    for cls in np.unique(labels):
        cidx = idx[labels == cls]
        rng.shuffle(cidx)
        parts.append(cidx)
    order = np.concatenate(parts)
    rng.shuffle(order)
    n_tr = max(1, int(round(n * train_size)))
    # ensure both classes in train and test when possible
    tr, te = order[:n_tr], order[n_tr:]
    if len(te) == 0:
        te = order[-1:]
        tr = order[:-1]
    if isinstance(features, dict):
        x_tr = {k: np.asarray(v)[tr] for k, v in features.items() if k != "labels"}
        x_te = {k: np.asarray(v)[te] for k, v in features.items() if k != "labels"}
        return x_tr, x_te, labels[tr], labels[te]
    feats = np.asarray(features)
    return feats[tr], feats[te], labels[tr], labels[te]
