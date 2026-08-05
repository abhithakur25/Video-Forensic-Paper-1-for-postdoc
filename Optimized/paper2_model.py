"""Paper 2's proposed model (BiLSTMGBM) applied to Paper 1's features.

Ported faithfully from CODE_05-08-2025_Paper2/SubFunctions/Model.py:

  * stacked Bidirectional LSTM (100, 128, 128) with ReLU + Dropout(0.5);
  * multi-level attention (mutual + spatial self-attention);
  * mixed attention (channel attention + zero attention);
  * Dense 64 -> Dense 32 -> softmax;
  * incremental learning over 5 cumulative chunks of the training set;
  * epochs = 500, batch_size = 32, Adam(lr=0.001)  -- Paper 2's settings;
  * the trained network used as a feature extractor, with a
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1) fitted on those features.

ONE STEP IS DELIBERATELY OMITTED
--------------------------------
Paper 2 calls, between training and feature extraction:

    model = Optimization(model, x_test, self.y_test).main_update_hyperparameters()

`Optimization` receives the test features and test labels, and searches the
model's weights to maximise the score computed on them
(Optimization.py:31-55). That is fitting the test set, and the score it
maximises is the tampered metric, so `HYBRID(epoch=10, pop_size=50)` is 500
draws from a random number generator with the best one kept -- which is where
the reported 100% comes from. Including it would produce a number that
measures nothing, so it is left out and everything below is scored with a real
confusion matrix.

INPUT ADAPTATION
----------------
Paper 2 reshapes its 5-D feature tensor (n, a, b, c, d) -> (n, a*b, c*d).
Applied literally to Paper 1's 'proposed' tensor (50, 10, 128, 128, 12) that
gives a 1280-step sequence of 1536 features, which is not trainable on CPU.
Instead each frame is reduced to a multi-scale grid-pooled descriptor, giving
a 10-step sequence of 252 features -- which preserves exactly the temporal
sequence structure the Bi-LSTM exists to model.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

P = Path(__file__).resolve().parents[1]
os.chdir(P)
sys.path.insert(0, str(P))
sys.path.insert(0, str(P / "Optimized"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
warnings.filterwarnings("ignore")

SEED = 1234
EPOCHS = int(os.environ.get("P2_EPOCHS", "500"))
BATCH = 32
LR = 0.001
CHUNKS = 5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def frame_descriptors(pr, grids=(1, 2, 4)):
    """(n, T, H, W, C) -> (n, T, D): multi-scale grid pooling per frame."""
    n, T, H, W, C = pr.shape
    out = np.zeros((n, T, sum(g * g for g in grids) * C), dtype=np.float32)
    for i in range(n):
        for t in range(T):
            v = []
            for g in grids:
                hs, ws = H // g, W // g
                for a in range(g):
                    for b in range(g):
                        v.append(pr[i, t, a*hs:(a+1)*hs, b*ws:(b+1)*ws].mean((0, 1)))
            out[i, t] = np.concatenate(v)
    return out


def increment_chunks(n, k=CHUNKS):
    """Paper 2's Incremental_Learning: cumulative prefixes of the training set."""
    idx = list(range(n))
    avg, rem = n // k, n % k
    parts, s = [], 0
    for i in range(k):
        sz = avg + (1 if i < rem else 0)
        parts.append(idx[s:s + sz])
        s += sz
    return [sum(parts[:i + 1], []) for i in range(k)]


def build_model(steps, feat, n_class):
    """Paper 2's BiLSTMGBM network. The attention blocks are reproduced with
    standard Keras layers: multi-head self-attention for the multi-level
    stage, and channel (squeeze-excite) attention for the mixed stage."""
    import tensorflow as tf
    from keras.layers import (Activation, Add, Bidirectional, Dense, Dropout,
                              GlobalAveragePooling1D, Input, LayerNormalization,
                              LSTM, MultiHeadAttention, Multiply, Reshape)
    from keras.models import Model

    inp = Input(shape=(steps, feat))
    x = Bidirectional(LSTM(100, return_sequences=True))(inp)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)

    # --- multi-level attention: mutual/self attention over the sequence
    a = LayerNormalization(epsilon=1e-6)(x)
    a = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(a, a)
    x = Add()([x, a])

    # --- mixed attention: channel (squeeze-excite) + residual
    s = GlobalAveragePooling1D()(x)
    s = Dense(x.shape[-1] // 8, activation="relu")(s)
    s = Dense(x.shape[-1], activation="sigmoid")(s)
    s = Reshape((1, x.shape[-1]))(s)
    x = Add()([x, Multiply()([x, s])])

    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, name="feat64")(x)
    x = Activation("relu")(x)
    x = Dense(32, name="feat32")(x)
    x = Activation("relu")(x)
    out = Dense(n_class, activation="softmax")(x)
    m = Model(inp, out)
    m.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              metrics=["accuracy"])
    return m


def fit_predict(Xtr, ytr, Xte, epochs=EPOCHS):
    import random
    import tensorflow as tf
    from keras.models import Model
    from keras.utils import to_categorical
    from sklearn.ensemble import GradientBoostingClassifier

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    # standardise with TRAINING statistics only
    mu = Xtr.mean((0, 1), keepdims=True)
    sd = Xtr.std((0, 1), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    model = build_model(Xtr.shape[1], Xtr.shape[2], 2)
    cw = {c: len(ytr) / (2 * max(1, int((ytr == c).sum()))) for c in (0, 1)}

    for ci, chunk in enumerate(increment_chunks(len(Xtr))):
        model.fit(Xtr[chunk], to_categorical(ytr[chunk], 2), epochs=epochs,
                  batch_size=BATCH, verbose=0, shuffle=True, class_weight=cw)

    # Paper 2 calls Optimization(model, x_test, y_test) here. Omitted: it fits
    # the test set and maximises the fabricated metric.

    extractor = Model(model.input, model.get_layer("feat64").output)
    ftr = extractor.predict(Xtr, verbose=0)
    fte = extractor.predict(Xte, verbose=0)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                    max_depth=1, random_state=0)
    gb.fit(ftr, ytr)
    nn = np.argmax(model.predict(Xte, verbose=0), axis=1)
    tf.keras.backend.clear_session()
    return gb.predict(fte), nn


def main():
    import pickle
    from metrics_fixed import balanced_accuracy, evaluation_metrics
    from sklearn.model_selection import StratifiedKFold

    with open(P / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    y = np.asarray(data["labels"]).astype(int)
    pr = np.asarray(data["proposed"], dtype=np.float32)

    log("building per-frame descriptors")
    X = frame_descriptors(pr)
    log(f"sequence tensor {X.shape}  (Paper 2 settings: {EPOCHS} epochs, "
        f"batch {BATCH}, lr {LR}, {CHUNKS} incremental chunks)")

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows_gb, rows_nn = [], []
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        pg, pn = fit_predict(X[tr], y[tr], X[te])
        mg = evaluation_metrics(y[te], pg) + [balanced_accuracy(y[te], pg)]
        mn = evaluation_metrics(y[te], pn) + [balanced_accuracy(y[te], pn)]
        rows_gb.append(mg); rows_nn.append(mn)
        log(f"  fold {k}: BiLSTM+GBM acc {mg[0]*100:6.2f} bal {mg[5]*100:6.2f} | "
            f"BiLSTM-only acc {mn[0]*100:6.2f} bal {mn[5]*100:6.2f} "
            f"[{time.time()-t0:.0f}s]")

    for name, rows in (("BiLSTMGBM (Paper 2 model)", rows_gb),
                       ("BiLSTM softmax only", rows_nn)):
        a = np.asarray(rows)
        log(f"{name}: acc {np.nanmean(a[:,0])*100:.2f}  sen "
            f"{np.nanmean(a[:,1])*100:.2f}  spe {np.nanmean(a[:,2])*100:.2f}  "
            f"pre {np.nanmean(a[:,3])*100:.2f}  f1 {np.nanmean(a[:,4])*100:.2f}"
            f"  BAL {np.nanmean(a[:,5])*100:.2f}")

    (P / "Optimized" / "paper2_model.json").write_text(json.dumps({
        "settings": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                     "chunks": CHUNKS, "protocol": "stratified 5-fold",
                     "omitted": "Optimization(model, x_test, y_test) - fits "
                                "the test set and maximises a fabricated metric"},
        "BiLSTMGBM": np.asarray(rows_gb).mean(0).tolist(),
        "BiLSTM_only": np.asarray(rows_nn).mean(0).tolist(),
        "per_fold_gb": np.asarray(rows_gb).tolist(),
    }, indent=2), encoding="utf-8")
    log("wrote Optimized/paper2_model.json")


if __name__ == "__main__":
    main()
