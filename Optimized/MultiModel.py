# -*- coding: utf-8 -*-
"""Trainable multi-model comparison for Paper 1 Features.pkl.

Models (Paper-2-style cohort + Paper-1 proposed):
  DCNN              — compact Conv2D on spatial projection of proposed features
  EfficientNetV2B0  — latest TF-2.10 keras.applications backbone
  MobileNetV2       — lightweight modern backbone
  STIDNet           — teacher–student distillation (Paper 1 comparative)
  P1-Proposed       — 3D-CNN + dual LSTM + SCAM + MUSE (ThreeDCNNLSTM opt=3)

All train/predict on the same index splits. Metrics must be scored with
Optimized.metrics_fixed (never mealpy.metrics).
"""
from __future__ import annotations

import numpy as np
from keras.layers import (
    Activation, Add, AveragePooling3D as AvgPool3D, BatchNormalization,
    Bidirectional, Conv1D, Conv2D, Conv3D, Dense, Dropout, Flatten, Input,
    LSTM, Lambda, LayerNormalization, MaxPooling1D as MaxPool1D,
    MaxPooling2D, MaxPooling3D, MultiHeadAttention, Reshape,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2

from Optimized.feature_adapters import (
    as_rgb_image, as_spatial_from_proposed, as_volume,
)
from Optimized.balance import keras_class_weight, oversample_minority

LATEST_BACKBONE = "EfficientNetV2B0"
LATEST_BACKBONE_REASON = (
    "EfficientNetV2B0 is the V2 successor to EfficientNetB0/B7 used in Paper 1 "
    "comparatives; available under TF 2.10 keras.applications without upgrade."
)


def _safe_pred(model, x_te, n_class):
    raw = model.predict(x_te, verbose=0)
    raw = np.asarray(raw)
    if raw.ndim == 1:
        # binary sigmoid single unit
        pred = (raw >= 0.5).astype(int)
    else:
        pred = np.argmax(raw, axis=1)
    return pred.astype(int)


def _fit_kw(class_weight):
    return {"class_weight": class_weight} if class_weight else {}


def train_predict_dcnn(x_prop, y_tr, x_prop_te, y_te, epochs=3, batch_size=8, lr=0.001,
                       size=32, class_weight=None, **_):
    x_tr = as_spatial_from_proposed(x_prop, size=size)
    x_te = as_spatial_from_proposed(x_prop_te, size=size)
    y_cat = to_categorical(y_tr)
    n_class = y_cat.shape[1]
    inp = Input(shape=x_tr.shape[1:])
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(n_class, activation="softmax")(x)
    model = Model(inp, out, name="DCNN_P1")
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(x_tr, y_cat, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True,
              **_fit_kw(class_weight))
    return _safe_pred(model, x_te, n_class)


def train_predict_efficientnetv2(x_prop, y_tr, x_prop_te, y_te, epochs=3,
                                 batch_size=4, lr=0.001, img_size=64, class_weight=None, **_):
    x_tr = as_rgb_image(x_prop, size=img_size) / 255.0
    x_te = as_rgb_image(x_prop_te, size=img_size) / 255.0
    y_cat = to_categorical(y_tr)
    n_class = y_cat.shape[1]
    base = EfficientNetV2B0(
        include_top=False, weights=None, input_shape=(img_size, img_size, 3), pooling="avg",
    )
    base.trainable = True
    inp = Input(shape=(img_size, img_size, 3))
    x = base(inp, training=True)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_class, activation="softmax")(x)
    model = Model(inp, out, name="EfficientNetV2B0_P1")
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(x_tr, y_cat, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True,
              **_fit_kw(class_weight))
    return _safe_pred(model, x_te, n_class)


def train_predict_mobilenetv2(x_prop, y_tr, x_prop_te, y_te, epochs=3,
                              batch_size=4, lr=0.001, img_size=64, class_weight=None, **_):
    x_tr = as_rgb_image(x_prop, size=img_size) / 255.0
    x_te = as_rgb_image(x_prop_te, size=img_size) / 255.0
    y_cat = to_categorical(y_tr)
    n_class = y_cat.shape[1]
    base = MobileNetV2(
        include_top=False, weights=None, input_shape=(img_size, img_size, 3), pooling="avg",
    )
    base.trainable = True
    inp = Input(shape=(img_size, img_size, 3))
    x = base(inp, training=True)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_class, activation="softmax")(x)
    model = Model(inp, out, name="MobileNetV2_P1")
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(x_tr, y_cat, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True,
              **_fit_kw(class_weight))
    return _safe_pred(model, x_te, n_class)


def _teacher(input_shape, n_class):
    inp = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation="relu")(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(n_class, activation="softmax")(x)
    return Model(inp, out, name="STIDNet_teacher")


def _student(input_shape, n_class):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu")(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(n_class, activation="softmax")(x)
    return Model(inp, out, name="STIDNet_student")


def train_predict_stidnet(x_prop, y_tr, x_prop_te, y_te, epochs=3, batch_size=8, lr=0.001,
                          size=32, class_weight=None, **_):
    """Paper-1-style teacher–student (distillation approximated by student fit after teacher)."""
    x_tr = as_spatial_from_proposed(x_prop, size=size)
    x_te = as_spatial_from_proposed(x_prop_te, size=size)
    y_cat = to_categorical(y_tr)
    n_class = y_cat.shape[1]
    shape = x_tr.shape[1:]
    teacher = _teacher(shape, n_class)
    student = _student(shape, n_class)
    teacher.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=["accuracy"])
    student.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=["accuracy"])
    teacher.fit(x_tr, y_cat, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True,
                **_fit_kw(class_weight))
    # soft targets from teacher
    soft = teacher.predict(x_tr, verbose=0)
    # blend hard + soft for student
    y_blend = 0.5 * y_cat + 0.5 * soft
    # sample_weight from class weights if provided
    sw = None
    if class_weight:
        sw = np.array([class_weight.get(int(yi), 1.0) for yi in y_tr], dtype=np.float32)
    student.fit(x_tr, y_blend, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True,
                sample_weight=sw)
    return _safe_pred(student, x_te, n_class)


def train_predict_p1_proposed(x_prop, y_tr, x_prop_te, y_te, epochs=3, batch_size=2,
                              lr=0.001, size=32, opt=3, class_weight=None, **_):
    """3D-CNN + dual LSTM + SCAM + MUSE (Paper 1 proposed ThreeDCNNLSTM, opt=3).

    Spatial dims downsampled to `size` for CPU feasibility; attention modules
    imported from SubFunctions when available, else skipped gracefully.
    """
    x_tr = as_volume(x_prop, size=size)
    x_te = as_volume(x_prop_te, size=size)
    y_cat = to_categorical(y_tr)
    n_class = y_cat.shape[1]

    # optional attention imports (Paper 1 SubFunctions)
    try:
        from SubFunctions.SCAM import SpatialAndChannelJointAttention
        has_scam = True
    except Exception:
        has_scam = False
        SpatialAndChannelJointAttention = None
    try:
        from SubFunctions.MUSE import multi_excited_block
        has_muse = True
    except Exception:
        has_muse = False
        multi_excited_block = None

    inp = Input(shape=x_tr.shape[1:])
    x = Conv3D(16, (3, 3, 3), padding="same")(inp)
    x = Activation("relu")(x)
    x1 = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x2 = AvgPool3D(pool_size=(1, 2, 2))(x)
    x = Lambda(lambda t: 0.5 * (t[0] + t[1]))([x1, x2])

    x = Conv3D(32, (3, 3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x1 = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x2 = AvgPool3D(pool_size=(1, 2, 2))(x)
    x = Lambda(lambda t: 0.5 * (t[0] + t[1]))([x1, x2])

    x = Conv3D(64, (3, 3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x1 = MaxPooling3D(pool_size=(1, 2, 2))(x)
    x2 = AvgPool3D(pool_size=(1, 2, 2))(x)
    x = Lambda(lambda t: 0.5 * (t[0] + t[1]))([x1, x2])
    x = Dropout(0.25)(x)

    # (B, T', H', W', C) -> (B, T'*H', W', C) then optional SCAM
    bshape = x.shape
    x = Reshape(target_shape=(int(bshape[1]) * int(bshape[2]), int(bshape[3]), int(bshape[4])))(x)
    if has_scam and (opt == 2 or opt == 3):
        try:
            x = SpatialAndChannelJointAttention()(x)
        except Exception:
            pass
    x = Reshape(target_shape=(int(x.shape[1]), int(x.shape[2]) * int(x.shape[3])))(x)

    x1 = LSTM(64, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal")(x)
    x2 = LSTM(64, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal")(x)
    x = Add()([x1, x2])

    if has_muse and (opt == 1 or opt == 3):
        try:
            x = multi_excited_block(x, int(x.shape[-1]), activation="elu",
                                    operation="average", dropprob=0.05)
        except Exception:
            pass

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(n_class, activation="softmax")(x)
    model = Model(inp, out, name="P1_Proposed_3DCNN_LSTM")
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(x_tr, y_cat, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True,
              **_fit_kw(class_weight))
    return _safe_pred(model, x_te, n_class)


def train_predict_rf(x_prop, y_tr, x_prop_te, y_te, epochs=3, class_weight=None,
                     seed=0, **_):
    """RandomForest on flattened time-mean features (strong on small N)."""
    from sklearn.ensemble import RandomForestClassifier
    from Optimized.feature_adapters import as_spatial_from_proposed, as_flat

    x_tr = as_flat(as_spatial_from_proposed(x_prop, size=32))
    x_te = as_flat(as_spatial_from_proposed(x_prop_te, size=32))
    cw = "balanced_subsample" if class_weight else None
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=1,
        class_weight=cw, random_state=seed, n_jobs=-1,
    )
    clf.fit(x_tr, np.asarray(y_tr).ravel())
    return clf.predict(x_te).astype(int)


def train_predict_gbm(x_prop, y_tr, x_prop_te, y_te, epochs=3, class_weight=None,
                      seed=0, **_):
    """GradientBoosting on flattened features; sample_weight from class_weight."""
    from sklearn.ensemble import GradientBoostingClassifier
    from Optimized.feature_adapters import as_spatial_from_proposed, as_flat

    x_tr = as_flat(as_spatial_from_proposed(x_prop, size=32))
    x_te = as_flat(as_spatial_from_proposed(x_prop_te, size=32))
    y_tr = np.asarray(y_tr).ravel()
    sw = None
    if class_weight:
        sw = np.array([class_weight.get(int(yi), 1.0) for yi in y_tr], dtype=np.float32)
    clf = GradientBoostingClassifier(
        n_estimators=min(300, max(50, int(epochs) * 5)),
        learning_rate=0.05, max_depth=3, random_state=seed,
    )
    clf.fit(x_tr, y_tr, sample_weight=sw)
    return clf.predict(x_te).astype(int)


MODEL_REGISTRY = {
    "DCNN": train_predict_dcnn,
    "EfficientNetV2B0": train_predict_efficientnetv2,
    "MobileNetV2": train_predict_mobilenetv2,
    "STIDNet": train_predict_stidnet,
    "P1-Proposed": train_predict_p1_proposed,
    "RF": train_predict_rf,
    "GBM": train_predict_gbm,
}


def run_model(name, x_tr, y_tr, x_te, y_te, epochs=3,
              use_class_weight=False, use_oversample=False, seed=0,
              minority_scale=1.0, oversample_ratio=1.0, **kwargs):
    """Train one model; optional balanced class weights + minority oversample (train only)."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name}; known={list(MODEL_REGISTRY)}")
    x_fit, y_fit = x_tr, y_tr
    if use_oversample:
        x_fit, y_fit = oversample_minority(
            x_fit, y_fit, seed=seed, ratio=oversample_ratio
        )
    cw = (
        keras_class_weight(y_fit, minority_scale=minority_scale)
        if use_class_weight
        else None
    )
    return MODEL_REGISTRY[name](
        x_fit, y_fit, x_te, y_te, epochs=epochs, class_weight=cw, seed=seed, **kwargs
    )
