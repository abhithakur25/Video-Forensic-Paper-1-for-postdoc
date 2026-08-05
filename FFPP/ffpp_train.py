"""Train a forgery detector on cached FaceForensics++ face crops.

Frame-level training, VIDEO-level evaluation. Produces the same tables as the
paper: metrics against training percentage (40-90%) and against k-fold, scored
with a real confusion matrix.

WHY THIS CAN REACH THE 90s AND THE 50-VIDEO PIPELINE CANNOT
-----------------------------------------------------------
The existing evaluation has 50 videos reduced to one feature vector each, so a
training split is 19-44 samples. Here, 1,000 originals + 1,000 manipulated at
32 crops per video is ~64,000 training images. That is the difference between
a model that memorises its training set and one that generalises, and it is
the only reason published FF++ detectors reach the low-to-mid 90s.

SPLIT INTEGRITY
---------------
Splits are grouped by source identity, not by frame and not by video. FF++
manipulated clips are named "<target>_<source>.mp4" and share footage with
"<target>.mp4"; if the original landed in train and the manipulation in test,
the model could match background and lighting rather than the manipulation.
GroupShuffleSplit on identity prevents both that and frame-level leakage.
Every run asserts the groups are disjoint and prints the check.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "Optimized"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 1234
BACKBONES = {
    "EfficientNetV2B0": ("efficientnet_v2", "EfficientNetV2B0"),
    "EfficientNetV2S": ("efficientnet_v2", "EfficientNetV2S"),
    "MobileNetV3Large": ("mobilenet_v3", "MobileNetV3Large"),
    "ConvNeXtTiny": ("convnext", "ConvNeXtTiny"),
    "Xception": ("xception", "Xception"),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_cache(cache):
    c = Path(cache)
    X = np.load(c / "frames.npy", mmap_mode="r")
    y = np.load(c / "labels.npy")
    v = np.load(c / "video_index.npy")
    ident = np.load(c / "identity_of_video.npy")
    meta = json.loads((c / "meta.json").read_text(encoding="utf-8"))
    groups = ident[v]                       # identity per frame
    log(f"{len(X)} crops from {meta['n_videos']} videos, "
        f"{len(np.unique(groups))} identities, "
        f"class balance {np.bincount(y).tolist()}")
    return X, y, v, groups, meta


def assert_disjoint(groups, tr, te, what):
    a, b = set(groups[tr].tolist()), set(groups[te].tolist())
    overlap = a & b
    if overlap:
        raise SystemExit(f"LEAKAGE in {what}: {len(overlap)} identities appear "
                         f"in both train and test")
    log(f"    split integrity OK ({what}): {len(a)} train / {len(b)} test "
        f"identities, 0 shared")


def build_model(name, size, lr, trainable_from=None):
    import importlib
    import tensorflow as tf
    from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from keras.models import Model

    mod_name, cls = BACKBONES[name]
    mod = importlib.import_module(f"keras.applications.{mod_name}")
    base = getattr(mod, cls)(weights="imagenet", include_top=False,
                             input_shape=(size, size, 3))
    if trainable_from is None:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:trainable_from]:
            layer.trainable = False

    inp = Input((size, size, 3))
    x = base(inp, training=False if trainable_from is None else None)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(2, activation="softmax")(x)
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m, mod


def make_ds(X, y, idx, pre, batch, size, training, seed=SEED):
    import tensorflow as tf

    def gen():
        order = np.array(idx)
        if training:
            rs = np.random.RandomState(seed)
            rs.shuffle(order)
        for i in order:
            yield np.asarray(X[i], np.float32), np.eye(2, dtype=np.float32)[y[i]]

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec((size, size, 3), tf.float32),
            tf.TensorSpec((2,), tf.float32)))
    if training:
        ds = ds.map(lambda a, b: (tf.image.random_flip_left_right(a), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda a, b: (tf.image.random_brightness(a, 12.0), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda a, b: (pre(a), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def video_level(prob, v, y, idx, how="mean"):
    """Aggregate frame probabilities into one decision per video."""
    vids = np.unique(v[idx])
    yt, yp = [], []
    for vid in vids:
        m = v[idx] == vid
        p = prob[m][:, 1]
        s = np.median(p) if how == "median" else p.mean()
        yt.append(int(y[idx][m][0]))
        yp.append(int(s >= 0.5))
    return np.array(yt), np.array(yp)


def run_split(X, y, v, groups, tr, te, args, tag):
    import tensorflow as tf
    from metrics_fixed import balanced_accuracy, evaluation_metrics

    assert_disjoint(groups, tr, te, tag)
    model, mod = build_model(args.backbone, args.size, args.lr,
                             None if args.freeze else args.trainable_from)
    pre = getattr(mod, "preprocess_input", lambda a: a)

    cw = {c: len(tr) / (2 * max(1, int((y[tr] == c).sum()))) for c in (0, 1)}
    dtr = make_ds(X, y, tr, pre, args.batch, args.size, True)
    dte = make_ds(X, y, te, pre, args.batch, args.size, False)

    model.fit(dtr, epochs=args.epochs, verbose=2, class_weight=cw)
    prob = model.predict(dte, verbose=0)

    fy, fp = y[te], np.argmax(prob, 1)
    frame_m = evaluation_metrics(fy, fp) + [balanced_accuracy(fy, fp)]
    vy, vp = video_level(prob, v, y, te, args.aggregate)
    vid_m = evaluation_metrics(vy, vp) + [balanced_accuracy(vy, vp)]

    log(f"    frame-level  acc {frame_m[0]*100:6.2f}  bal {frame_m[5]*100:6.2f}"
        f"   ({len(te)} crops)")
    log(f"    VIDEO-level  acc {vid_m[0]*100:6.2f}  bal {vid_m[5]*100:6.2f}"
        f"   ({len(vy)} videos)")
    tf.keras.backend.clear_session()
    return frame_m, vid_m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="FFPP/cache")
    ap.add_argument("--backbone", default="EfficientNetV2B0",
                    choices=list(BACKBONES))
    ap.add_argument("--mode", default="tp", choices=["tp", "kfold", "single"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--freeze", action="store_true",
                    help="frozen backbone, head only (fast baseline)")
    ap.add_argument("--trainable-from", type=int, default=-40,
                    help="fine-tune the last N layers (negative index)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--aggregate", default="mean", choices=["mean", "median"])
    ap.add_argument("--out", default="FFPP/results")
    args = ap.parse_args()

    from sklearn.model_selection import GroupShuffleSplit, GroupKFold

    X, y, v, groups, meta = load_cache(args.cache)
    outdir = P / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}

    if args.mode == "single":
        gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
        tr, te = next(gss.split(X, y, groups))
        f, vd = run_split(X, y, v, groups, tr, te, args, "single 80/20")
        results["single"] = {"frame": f, "video": vd}

    elif args.mode == "tp":
        for pct in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            log(f"=== training percentage {pct:.0%}")
            gss = GroupShuffleSplit(1, train_size=pct, random_state=SEED)
            tr, te = next(gss.split(X, y, groups))
            f, vd = run_split(X, y, v, groups, tr, te, args, f"tp {pct:.0%}")
            results[f"{int(pct*100)}%"] = {"frame": f, "video": vd}
            save(results, args, outdir)

    else:
        gkf = GroupKFold(n_splits=args.folds)
        for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
            log(f"=== fold {k}/{args.folds}")
            f, vd = run_split(X, y, v, groups, tr, te, args, f"fold {k}")
            results[f"fold{k}"] = {"frame": f, "video": vd}
            save(results, args, outdir)

    save(results, args, outdir)
    log("")
    log(f"{'split':<10}{'frame acc':>11}{'frame bal':>11}"
        f"{'VIDEO acc':>11}{'VIDEO bal':>11}")
    for k, r in results.items():
        log(f"{k:<10}{r['frame'][0]*100:10.2f} {r['frame'][5]*100:10.2f} "
            f"{r['video'][0]*100:10.2f} {r['video'][5]*100:10.2f}")
    va = np.mean([r["video"][0] for r in results.values()]) * 100
    vb = np.mean([r["video"][5] for r in results.values()]) * 100
    log(f"{'MEAN':<10}{'':>22}{va:10.2f} {vb:10.2f}")


def save(results, args, outdir):
    (outdir / f"{args.backbone}_{args.mode}.json").write_text(json.dumps({
        "backbone": args.backbone, "mode": args.mode, "epochs": args.epochs,
        "batch": args.batch, "lr": args.lr, "size": args.size,
        "freeze": args.freeze, "aggregate": args.aggregate,
        "scored_with": "Optimized/metrics_fixed.py",
        "columns": ["ACC", "SEN", "SPE", "PRE", "F1", "BAL"],
        "split_grouping": "by source identity - no frame or identity leakage",
        "results": {k: {"frame": list(map(float, r["frame"])),
                        "video": list(map(float, r["video"]))}
                    for k, r in results.items()},
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
