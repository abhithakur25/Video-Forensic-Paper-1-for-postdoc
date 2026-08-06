"""Optimised re-evaluation of the Paper 1 video-forgery pipeline.

Two independent tracks, both written so their numbers drop straight into the
same tables as the published run (metrics come from the authors' own
SubFunctions.Evaluate.Evaluation_Metrics, splits from the same deterministic
per-class prefix rule as SubFunctions.Analysis).

Track A - LATEST-GENERATION BACKBONES
    The paper's "EfficientNet" comparison is EfficientNetB7 truncated to
    base_model.layers[:6] (rescaling, normalisation, zero-pad, stem conv) with
    two fresh Conv2D layers bolted on, and it is fed [0, 1] data even though
    the EfficientNet family expects [0, 255].  This track runs the current
    generation of the same families properly: EfficientNetV2, ConvNeXt,
    MobileNetV3 and ResNet-RS, each as a frozen ImageNet feature extractor
    with a small trained head.

    Frozen, not fine-tuned, on purpose: a training split here is 19-44 samples
    and these backbones carry 5-25 M parameters.  Fine-tuning would memorise
    the split.  Embeddings are therefore computed once for all 50 samples and
    cached, and only the head is refit per split - which is also what makes
    the sweep cheap.

Track B - OPTIMISED SMA-CLMPNet
    Same architecture as the published SMA-CLMPNet: the authors' own MUSE
    multi_excited_block and SCAM SpatialAndChannelJointAttention, the same
    3D convolution stack, the same dual-LSTM.  Only the training recipe
    changes, so the comparison isolates the recipe:

      1. batch_size 32 -> 8.  With 44 training samples, batch 32 gives two
         gradient steps per epoch; the published 10-epoch budget is therefore
         20 weight updates in total.
      2. Inputs standardised with statistics taken from the TRAINING split
         only.  The 'proposed' tensor spans -24.8 to 255 unnormalised.
      3. Class weights.  The corpus is 29/21, and every metric in the table
         is computed against class 0.
      4. Cosine-decayed learning rate over a longer budget instead of a flat
         1e-3.  No early stopping: with 19 training samples at the 40% split
         there is no honest validation set to stop on, so the budget is fixed
         in advance and identical for every split.

Nothing in SubFunctions/ is modified - this module imports it.
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "Optimized"))

# SubFunctions/Model.py prints emoji through termcolor.cprint.  When stdout is
# redirected to a file Python picks cp1252 here and those calls raise
# UnicodeEncodeError, killing the run several minutes in.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SEED = 1234
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# The five comparison feature sets, in the order the paper reports them.
NAMES_PUBLISHED = ["EfficientNet", "STIDNet", "DCNN", "GLCM", "BA-TFD",
                   "MUSE-CLMPNet", "SCAM-CLMPNet", "SMA-CLMPNet"]

# Track A: current-generation backbones, paired with the paper's generation.
LATEST = {
    "EfficientNetV2S":  ("efficientnet_v2", "EfficientNetV2S",  384,
                         "supersedes the paper's EfficientNetB7 stem"),
    "ConvNeXtTiny":     ("convnext",        "ConvNeXtTiny",     224,
                         "2022 pure-convolutional design"),
    "MobileNetV3Large": ("mobilenet_v3",    "MobileNetV3Large", 224,
                         "efficiency-oriented current generation"),
    "ResNetRS50":       ("resnet_rs",       "ResNetRS50",       160,
                         "re-scaled ResNet, supersedes ResNet-50"),
}

# The paper's own models, re-run so their REAL scores can be compared against
# the latest backbones on equal terms.  BA-TFD is absent: its ViTDCNN applies
# MaxPooling2D(1, 1), which does not downsample, so Dense(2048) needs an 8.6 GB
# weight matrix and exhausts memory at every batch size.
PUBLISHED_MODELS = {
    "EfficientNet":  lambda n: n.EfficientNet(),
    "STIDNet":       lambda n: n.STIDNet(),
    "DCNN":          lambda n: n.CNN(),
    "GLCM":          lambda n: n.GLCM(),
    "MUSE-CLMPNet":  lambda n: n.ThreeDCNNLSTM(opt=1),
    "SCAM-CLMPNet":  lambda n: n.ThreeDCNNLSTM(opt=2),
    "SMA-CLMPNet":   lambda n: n.ThreeDCNNLSTM(opt=3),
}

TRAIN_PCTS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def subfunctions_lite():
    """Register SubFunctions as a namespace package without running its
    __init__.py, which imports torch.  torch and the conda-forge scipy in this
    environment ship colliding copies of libiomp5md.dll and the process dies
    on import."""
    if "SubFunctions" in sys.modules:
        return
    pkg = types.ModuleType("SubFunctions")
    pkg.__path__ = [str(PROJECT / "SubFunctions")]
    pkg.__package__ = "SubFunctions"
    sys.modules["SubFunctions"] = pkg


def seed_everything():
    import random
    import tensorflow as tf
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


# --------------------------------------------------------------------- data
def load_features():
    with open(PROJECT / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    log(f"features loaded: proposed{np.shape(data['proposed'])} "
        f"comparative1{np.shape(data['comparative1'])} "
        f"labels{np.shape(data['labels'])}")
    return data


def split_indices(labels, train_size):
    """Reproduce SubFunctions.Analysis exactly: per class, take the first
    train_size fraction of that class's indices as training.  It is a
    deterministic prefix split, not a random one, so it is reproducible
    without seeding."""
    labels = np.asarray(labels)
    tr, te = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        cut = int(len(idx) * train_size)
        tr.extend(idx[:cut])
        te.extend(idx[cut:])
    return np.array(tr), np.array(te)


def class_weights(y):
    y = np.asarray(y).astype(int)
    n, k = len(y), len(np.unique(y))
    return {int(c): n / (k * int(np.sum(y == c))) for c in np.unique(y)}


# ------------------------------------------------------------------ track A
def embed(name, data, cache_dir):
    """Compute (and cache) frozen ImageNet embeddings for all 50 samples."""
    cache = cache_dir / f"emb_{name}.npy"
    if cache.exists():
        e = np.load(cache)
        log(f"  {name}: embeddings from cache {e.shape}")
        return e

    import cv2
    import importlib
    import tensorflow as tf

    mod_name, cls_name, size, _ = LATEST[name]
    mod = importlib.import_module(f"keras.applications.{mod_name}")
    ctor = getattr(mod, cls_name)
    pre = getattr(mod, "preprocess_input", None)

    log(f"  {name}: building frozen backbone at {size}x{size} "
        f"(downloads ImageNet weights on first use)")
    backbone = ctor(weights="imagenet", include_top=False,
                    input_shape=(size, size, 3), pooling="avg")
    backbone.trainable = False

    # Match the authors' channel choice - they take [:, :, :3] of the
    # comparative tensor - but feed the [0, 255] range these families expect
    # instead of the [0, 1] the published code passes in.
    src = np.asarray(data["comparative1"])
    imgs = np.stack([cv2.resize(src[i][:, :, :3].astype(np.float32),
                                (size, size)) for i in range(src.shape[0])])
    imgs *= 255.0
    if pre is not None:
        imgs = pre(imgs)

    e = backbone.predict(imgs, batch_size=4, verbose=0)
    e = np.asarray(e, dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, e)
    log(f"  {name}: embeddings {e.shape} cached "
        f"({backbone.count_params()/1e6:.1f} M frozen params)")
    del backbone
    # NOT clear_session(): it resets Keras's global name counter, so the
    # next Input() is named 'input_1' again and collides with the cached
    # InputLayer of the module-level EfficientNetB7 in SubFunctions/Model.py
    # ('The name "input_1" is used 2 times in the model').
    gc.collect()
    return e


def fit_head(emb, y_tr, y_te, tr, te, epochs=200):
    """Train the small classifier head on precomputed embeddings."""
    import tensorflow as tf
    from keras.layers import Dense, Dropout, Input
    from keras.models import Model
    from keras.utils import to_categorical

    seed_everything()
    x_tr, x_te = emb[tr], emb[te]
    mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd

    inp = Input(shape=(emb.shape[1],))
    h = Dropout(0.3)(inp)
    h = Dense(64, activation="relu")(h)
    h = Dropout(0.3)(h)
    out = Dense(2, activation="softmax")(h)
    model = Model(inp, out)

    steps = max(1, int(np.ceil(len(tr) / 8))) * epochs
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, steps)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(x_tr, to_categorical(y_tr, 2), epochs=epochs, batch_size=8,
              verbose=0, shuffle=True, class_weight=class_weights(y_tr))
    yhat = np.argmax(model.predict(x_te, verbose=0), axis=1)
    # NOT clear_session(): it resets Keras's global name counter, so the
    # next Input() is named 'input_1' again and collides with the cached
    # InputLayer of the module-level EfficientNetB7 in SubFunctions/Model.py
    # ('The name "input_1" is used 2 times in the model').
    gc.collect()
    return yhat


# ------------------------------------------------------------------ track B
def build_smaclmpnet(input_shape, n_class, opt=3):
    """The published SMA-CLMPNet, layer for layer (SubFunctions/Model.py,
    ThreeDCNNLSTM).  Architecture unchanged - only how it is trained differs."""
    import tensorflow as tf
    from keras.layers import (Activation, Add, AvgPool3D, BatchNormalization,
                              Conv3D, Dense, Dropout, Flatten, Input, Lambda,
                              LSTM, MaxPooling3D, Reshape)
    from keras.models import Model
    from SubFunctions.MUSE import multi_excited_block
    from SubFunctions.SCAM import SpatialAndChannelJointAttention

    input_layer = Input(shape=input_shape)

    x = Conv3D(16, (3, 3, 3), padding="same")(input_layer)
    x = Activation("relu")(x)
    x1, x2 = MaxPooling3D(1, 1)(x), AvgPool3D(1, 1)(x)
    x = Lambda(lambda t: tf.reduce_mean(t, axis=0))([x1, x2])

    x = Conv3D(32, (3, 3, 3), padding="valid")(x)
    x = Activation("relu")(x)
    x1, x2 = MaxPooling3D(1, 2)(x), AvgPool3D(1, 2)(x)
    x = Lambda(lambda t: tf.reduce_mean(t, axis=0))([x1, x2])

    x = Conv3D(64, (3, 3, 3), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x1, x2 = MaxPooling3D(1, 2)(x), AvgPool3D(1, 2)(x)
    x = Lambda(lambda t: tf.reduce_mean(t, axis=0))([x1, x2])
    x = Dropout(0.25)(x)
    x = Reshape((x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))(x)

    if opt in (2, 3):
        x = SpatialAndChannelJointAttention()(x)
    x = Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)

    x1 = LSTM(128, kernel_initializer="glorot_uniform",
              recurrent_initializer="orthogonal")(x)
    x2 = LSTM(128, kernel_initializer="glorot_uniform",
              recurrent_initializer="orthogonal")(x)
    x = Add()([x1, x2])

    if opt in (1, 3):
        x = multi_excited_block(x, x.shape[-1], activation="elu",
                                operation="average", dropprob=0.05)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, name="dense2c")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(n_class, activation="softmax")(x)
    return Model(input_layer, out)


def fit_smaclmpnet(data, tr, te, epochs, batch_size, opt=3, normalise=True):
    import tensorflow as tf
    from keras.utils import to_categorical

    seed_everything()
    prop = _float32_cache(data)[5]
    y = np.asarray(data["labels"]).astype(int)
    x_tr, x_te = prop[tr].copy(), prop[te].copy()

    if normalise:
        # Training statistics only - the test split must not inform them.
        mu = x_tr.mean(axis=(0, 1, 2, 3), keepdims=True)
        sd = x_tr.std(axis=(0, 1, 2, 3), keepdims=True) + 1e-6
        x_tr = (x_tr - mu) / sd
        x_te = (x_te - mu) / sd

    model = build_smaclmpnet(x_tr.shape[1:], 2, opt=opt)
    steps = max(1, int(np.ceil(len(tr) / batch_size))) * epochs
    lr = tf.keras.optimizers.schedules.CosineDecay(1e-3, steps)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(x_tr, to_categorical(y[tr], 2), epochs=epochs,
              batch_size=batch_size, verbose=2, shuffle=True,
              class_weight=class_weights(y[tr]))
    yhat = np.argmax(model.predict(x_te, verbose=0), axis=1)
    # NOT clear_session(): it resets Keras's global name counter, so the
    # next Input() is named 'input_1' again and collides with the cached
    # InputLayer of the module-level EfficientNetB7 in SubFunctions/Model.py
    # ('The name "input_1" is used 2 times in the model').
    gc.collect()
    return yhat


_F32 = {}


def _float32_cache(data):
    """The six feature tensors as float32, converted once per process."""
    if not _F32:
        for k in ["comparative1", "comparative2", "comparative3",
                  "comparative4", "comparative5", "proposed"]:
            _F32[k] = np.asarray(data[k], dtype=np.float32)
        total = sum(a.nbytes for a in _F32.values()) / 1e9
        log(f"  float32 feature cache built ({total:.2f} GB)")
    return [_F32[k] for k in ["comparative1", "comparative2", "comparative3",
                              "comparative4", "comparative5", "proposed"]]


def fit_published(name, data, tr, te, epochs):
    """Run one of the paper's own models, unmodified, and return its real
    predictions.  Only the scoring of those predictions was ever broken - the
    models themselves train normally, so SubFunctions/Model.py is used as-is."""
    import tensorflow as tf
    from SubFunctions.Model import Network

    seed_everything()
    # Converted once and cached: 'proposed' alone is 786 MB as float64, and
    # this function is called 42 times over a full sweep.
    arrs = _float32_cache(data)
    y = np.asarray(data["labels"]).astype(int)

    net = Network(*[a[tr] for a in arrs], *[a[te] for a in arrs],
                  y[tr], y[te], epochs)
    yhat = PUBLISHED_MODELS[name](net)
    del net
    # NOT clear_session(): it resets Keras's global name counter, so the
    # next Input() is named 'input_1' again and collides with the cached
    # InputLayer of the module-level EfficientNetB7 in SubFunctions/Model.py
    # ('The name "input_1" is used 2 times in the model').
    gc.collect()
    return np.asarray(yhat).astype(int)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["diag", "sweep", "embed", "kfold"],
                    default="sweep")
    ap.add_argument("--ks", default="6,7,8,9,10",
                    help="kfold mode: k values to evaluate")
    ap.add_argument("--folds-per-k", type=int, default=1,
                    help="kfold mode: folds averaged per k")
    ap.add_argument("--epochs", type=int, default=40,
                    help="SMA-CLMPNet-Opt budget (published run used 10)")
    ap.add_argument("--head-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="published run used 32, i.e. 2 steps/epoch")
    ap.add_argument("--train-pct", type=float, default=0.9,
                    help="diag mode only")
    ap.add_argument("--models", default="",
                    help="comma-separated subset; default is all")
    ap.add_argument("--epochs-baseline", type=int, default=10,
                    help="budget for the paper's own models; 10 matches the "
                         "reproduction so only the scoring differs")
    ap.add_argument("--resume", action="store_true",
                    help="reuse splits already measured in --out")
    ap.add_argument("--out", default="Analysis1/OPT")
    args = ap.parse_args()

    subfunctions_lite()
    # NOT SubFunctions.Evaluate.Evaluation_Metrics: that routes through the
    # tampered mealpy._check_targets, which discards the predictions and
    # fabricates them. See Optimized/metrics_fixed.py for the evidence.
    from metrics_fixed import balanced_accuracy, evaluation_metrics

    data = load_features()
    labels = np.asarray(data["labels"]).astype(int)
    cache_dir = PROJECT / "Optimized" / "cache"

    wanted = [m for m in (args.models.split(",") if args.models else [])
              if m] or (list(PUBLISHED_MODELS) + list(LATEST)
                        + ["SMA-CLMPNet-Opt"])
    unknown = [w for w in wanted if w not in PUBLISHED_MODELS
               and w not in LATEST and w != "SMA-CLMPNet-Opt"]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}")

    # Embeddings are split-independent, so compute them once up front.
    embs = {}
    for name in [w for w in wanted if w in LATEST]:
        embs[name] = embed(name, data, cache_dir)
    if args.mode == "embed":
        log("embeddings only - done")
        return

    if args.mode == "kfold":
        return run_kfold(args, wanted, data, labels, embs,
                         evaluation_metrics, balanced_accuracy)

    pcts = [args.train_pct] if args.mode == "diag" else TRAIN_PCTS
    results = {name: [] for name in wanted}
    done_pcts = []

    # Resume: a completed split is expensive (~30 min) and already on disk.
    outdir = PROJECT / args.out
    man_path = outdir / "run_manifest.json"
    if args.resume and man_path.exists():
        prev = json.loads(man_path.read_text(encoding="utf-8"))
        prev_pcts = [float(p) for p in prev["train_pcts"]]
        if all((outdir / f"{n}.npy").exists() for n in wanted):
            for n in wanted:
                results[n] = [list(r) for r in np.load(outdir / f"{n}.npy")]
            keep = min(len(results[n]) for n in wanted)
            for n in wanted:
                results[n] = results[n][:keep]
            done_pcts = prev_pcts[:keep]
            log(f"resuming: {keep} split(s) already measured "
                f"({', '.join(f'{p:.0%}' for p in done_pcts)})")
        else:
            log("resume requested but not every model has an array - "
                "starting clean")

    t0 = time.time()

    for pct in pcts:
        if any(abs(pct - d) < 1e-9 for d in done_pcts):
            continue
        tr, te = split_indices(labels, pct)
        log(f"=== train {pct:.0%}: {len(tr)} train / {len(te)} test "
            f"(test classes {np.bincount(labels[te], minlength=2).tolist()})")
        for name in wanted:
            t1 = time.time()
            if name in LATEST:
                yhat = fit_head(embs[name], labels[tr], labels[te], tr, te,
                                epochs=args.head_epochs)
            elif name in PUBLISHED_MODELS:
                yhat = fit_published(name, data, tr, te, args.epochs_baseline)
            else:
                yhat = fit_smaclmpnet(data, tr, te, args.epochs,
                                      args.batch_size)
            m = evaluation_metrics(labels[te], yhat)
            bacc = balanced_accuracy(labels[te], yhat)
            results[name].append(m + [bacc])
            log(f"    {name:<18} acc {m[0]*100:6.2f}  bal-acc {bacc*100:6.2f}  "
                f"sen {m[1]*100:6.2f}  spe {m[2]*100:6.2f}  "
                f"f1 {m[4]*100:6.2f}   [{time.time()-t1:.0f}s]")

        # Checkpoint after every split: a killed process then costs one split,
        # not the whole sweep.
        done_pcts.append(pct)
        save(results, args, done_pcts, args.out)

    log(f"sweep finished in {time.time()-t0:.0f}s")
    save(results, args, done_pcts, args.out)


def run_kfold(args, wanted, data, labels, embs, metrics_fn, bal_fn):
    """Stratified k-fold with correct scoring, for section 5.6.2.

    The published KFAnalysis cannot be used: SubFunctions/Analysis.py:355
    indexes data['image'], a key ReadDataset never stores, and its scores go
    through the tampered metric anyway.
    """
    from sklearn.model_selection import StratifiedKFold

    ks = [int(k) for k in args.ks.split(",") if k]
    grid = {n: [] for n in wanted}
    done = []
    t0 = time.time()

    # Resume: one k value costs ~45 min and is checkpointed on completion.
    # Losing the whole run to a crash in the fifth k is not acceptable.
    outdir = PROJECT / args.out
    man_path = outdir / "run_manifest.json"
    if args.resume and man_path.exists():
        prev = json.loads(man_path.read_text(encoding="utf-8"))
        prev_ks = [int(k) for k in prev["k_values"]]
        if all((outdir / f"{n}.npy").exists() for n in wanted):
            for n in wanted:
                grid[n] = [list(r) for r in np.load(outdir / f"{n}.npy")]
            keep = min(len(grid[n]) for n in wanted)
            for n in wanted:
                grid[n] = grid[n][:keep]
            done = prev_ks[:keep]
            log(f"resuming: k = {done} already measured")
        else:
            log("resume requested but not every model has an array - "
                "starting clean")

    for k in ks:
        if k in done:
            continue
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        per_model = {n: [] for n in wanted}
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(labels)),
                                                  labels)):
            if fold >= args.folds_per_k:
                break
            log(f"=== k={k} fold {fold+1}/{args.folds_per_k}: {len(tr)} train "
                f"/ {len(te)} test "
                f"(test classes {np.bincount(labels[te], minlength=2).tolist()})")
            for name in wanted:
                t1 = time.time()
                if name in LATEST:
                    yhat = fit_head(embs[name], labels[tr], labels[te], tr, te,
                                    epochs=args.head_epochs)
                elif name in PUBLISHED_MODELS:
                    yhat = fit_published(name, data, tr, te,
                                         args.epochs_baseline)
                else:
                    yhat = fit_smaclmpnet(data, tr, te, args.epochs,
                                          args.batch_size)
                m = metrics_fn(labels[te], yhat)
                per_model[name].append(m + [bal_fn(labels[te], yhat)])
                log(f"    {name:<18} acc {m[0]*100:6.2f}  "
                    f"bal-acc {per_model[name][-1][5]*100:6.2f}  "
                    f"f1 {m[4]*100:6.2f}   [{time.time()-t1:.0f}s]")
        for name in wanted:
            grid[name].append(np.nanmean(np.asarray(per_model[name]), axis=0))
        done.append(k)
        save_kfold(grid, args, done, args.out)   # checkpoint after every k
        log(f"  [checkpoint] k values done: {done}")

    log(f"k-fold finished in {time.time()-t0:.0f}s")


def save_kfold(grid, args, ks_done, out):
    outdir = PROJECT / out
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in grid.items():
        if rows:
            np.save(outdir / f"{name}.npy", np.asarray(rows, dtype=float))
    (outdir / "run_manifest.json").write_text(json.dumps({
        "produced_by": "Optimized/optimize_models.py --mode kfold",
        "models": list(grid),
        "k_values": list(ks_done),
        "folds_per_k": args.folds_per_k,
        "epochs": args.epochs,
        "epochs_baseline": args.epochs_baseline,
        "head_epochs": args.head_epochs,
        "batch_size": args.batch_size,
        "scored_with": "Optimized/metrics_fixed.py (real confusion matrix)",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")


def save(results, args, pcts_done, out):
    outdir = PROJECT / out
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in results.items():
        if rows:
            np.save(outdir / f"{name}.npy", np.asarray(rows, dtype=float))
    (outdir / "run_manifest.json").write_text(json.dumps({
        "produced_by": "Optimized/optimize_models.py",
        "models": list(results),
        "train_pcts": [float(p) for p in pcts_done],
        "epochs": args.epochs,
        "epochs_baseline": args.epochs_baseline,
        "head_epochs": args.head_epochs,
        "scored_with": "Optimized/metrics_fixed.py (real confusion matrix)",
        "batch_size": args.batch_size,
        "mode": args.mode,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
