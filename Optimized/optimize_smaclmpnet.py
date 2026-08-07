"""Search SMA-CLMPNet's training recipe under nested cross-validation.

WHAT THIS SEARCHES
------------------
The published architecture is left exactly as it is - this imports
build_smaclmpnet from optimize_models rather than redefining it, so the MUSE
block, SCAM attention, 3D convolution stack and dual LSTM are the authors'.
What varies is the training recipe: batch size, epoch budget, initial learning
rate, the learning-rate schedule, class weighting, and input standardisation.

--ablate additionally varies `opt`, which toggles which of the two published
attention blocks are built (1 = MUSE only, 2 = SCAM only, 3 = both, the
published model). That is an architecture ablation, not a recipe change, and
results carry the distinction so it cannot be quietly reported as "tuning".

WHY NESTED CV, AND WHAT THE NUMBERS MEAN
----------------------------------------
Picking the best of N configurations on the same data you then report is how a
50-video corpus manufactures a good-looking number: with enough configurations,
one of them fits the noise. So selection happens strictly inside the training
folds. Each outer fold selects a configuration on its own inner folds, refits,
and predicts the outer test fold, which the selection never saw.

Two scores are therefore reported and they are not interchangeable:

    inner_best_mean   mean of each outer fold's best INNER score. Optimistic.
                      This is the number a non-nested search would print.
    nested_balanced   pooled out-of-fold balanced accuracy. Honest.

The gap between them is the selection bias, and on a corpus this small it is
usually large. Only nested_balanced belongs in the paper.

WHAT TO EXPECT
--------------
As measured in Optimized/RESULTS.md, SMA-CLMPNet scores exactly 50.00% balanced
accuracy at every training percentage, which on this 29/21 corpus means it
predicts one class for every input. It carries 2,258,534 parameters against 40
to 44 training samples. Every previous search here - the SMA-CLMPNet-Opt recipe,
30 class-weight and threshold configurations in optimize_weights.py, and the
higher-order feature work in optimize_v3.py - failed to beat an untuned L1
logistic regression at 77.17% nested balanced accuracy. A search that reports
"no configuration beats chance" is a real result and is written out as such.

USAGE
-----
    python Optimized/optimize_smaclmpnet.py --probe          # time one fit
    python Optimized/optimize_smaclmpnet.py --budget 12 --outer 5 --inner 3
    python Optimized/optimize_smaclmpnet.py --resume         # continue
"""
import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "Optimized"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 1234
OUT = PROJECT / "Optimized" / "smaclmpnet_search.json"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------- space
# Values are anchored on what the project has already established: the
# published recipe is batch 32 / 10 epochs / flat 1e-3, and SMA-CLMPNet-Opt is
# batch 8 / 30 epochs / cosine 1e-3 with class weights and standardisation.
RECIPE = {
    "batch_size": [4, 8, 16],
    "epochs": [30, 60],
    "lr0": [3e-4, 1e-3, 3e-3],
    "schedule": ["cosine", "flat"],
    "class_weight": [True, False],
    "normalise": [True, False],
}
ABLATE = {"opt": [1, 2, 3]}

PUBLISHED = {"batch_size": 32, "epochs": 10, "lr0": 1e-3, "schedule": "flat",
             "class_weight": False, "normalise": False, "opt": 3}
OPT_RECIPE = {"batch_size": 8, "epochs": 30, "lr0": 1e-3, "schedule": "cosine",
              "class_weight": True, "normalise": True, "opt": 3}


def build_space(ablate, budget, seed=SEED):
    space = dict(RECIPE)
    space.update(ABLATE if ablate else {"opt": [3]})
    keys = sorted(space)
    grid = [dict(zip(keys, v)) for v in itertools.product(*(space[k]
                                                           for k in keys))]
    # The two recipes already reported in the paper are always evaluated, so
    # the search is a superset of what is published and the comparison is
    # like-for-like rather than against a configuration nothing else used.
    anchors = [dict(PUBLISHED), dict(OPT_RECIPE)]
    if not ablate:
        anchors = [{**a, "opt": 3} for a in anchors]
    for a in anchors:
        if a not in grid:
            grid.append(a)
    if budget and budget < len(grid):
        rng = random.Random(seed)
        rest = [c for c in grid if c not in anchors]
        grid = anchors + rng.sample(rest, budget - len(anchors))
    return grid


# --------------------------------------------------------------------- fit
def fit_once(prop, y, tr, te, cfg):
    """One fit + prediction. Architecture untouched; recipe from cfg."""
    import tensorflow as tf
    from keras.utils import to_categorical
    from optimize_models import (build_smaclmpnet, class_weights,
                                 seed_everything)

    seed_everything()
    x_tr, x_te = prop[tr].copy(), prop[te].copy()
    if cfg["normalise"]:
        # Training statistics only - the test split must not inform them.
        mu = x_tr.mean(axis=(0, 1, 2, 3), keepdims=True)
        sd = x_tr.std(axis=(0, 1, 2, 3), keepdims=True) + 1e-6
        x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd

    model = build_smaclmpnet(x_tr.shape[1:], 2, opt=cfg["opt"])
    if cfg["schedule"] == "cosine":
        steps = max(1, int(np.ceil(len(tr) / cfg["batch_size"]))) * cfg["epochs"]
        lr = tf.keras.optimizers.schedules.CosineDecay(cfg["lr0"], steps)
    else:
        lr = cfg["lr0"]
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])
    model.fit(x_tr, to_categorical(y[tr], 2), epochs=cfg["epochs"],
              batch_size=cfg["batch_size"], verbose=0, shuffle=True,
              class_weight=class_weights(y[tr]) if cfg["class_weight"] else None)
    yhat = np.argmax(model.predict(x_te, verbose=0), axis=1)
    # NOT clear_session(): it resets Keras's global name counter and the next
    # Input() collides with the cached InputLayer in SubFunctions/Model.py.
    import gc
    gc.collect()
    return yhat


# ---------------------------------------------------------------- protocol
def nested_search(prop, y, grid, n_outer, n_inner, state):
    from sklearn.model_selection import StratifiedKFold
    from metrics_fixed import balanced_accuracy

    outer = StratifiedKFold(n_outer, shuffle=True, random_state=SEED)
    folds = list(outer.split(np.zeros(len(y)), y))

    oof = state.setdefault("oof", {})
    per_fold = state.setdefault("per_fold", [])
    done = {int(k) for k in oof}

    for fi, (tr, te) in enumerate(folds):
        if fi in done:
            log(f"outer fold {fi + 1}/{n_outer}: already done, skipping")
            continue
        t0 = time.time()
        inner = StratifiedKFold(n_inner, shuffle=True, random_state=SEED + fi)
        isplits = list(inner.split(np.zeros(len(tr)), y[tr]))

        scores = []
        for ci, cfg in enumerate(grid, 1):
            s = []
            for itr, ite in isplits:
                yhat = fit_once(prop, y, tr[itr], tr[ite], cfg)
                s.append(balanced_accuracy(y[tr[ite]], yhat) * 100)
            scores.append(float(np.mean(s)))
            log(f"  fold {fi + 1} cfg {ci}/{len(grid)} "
                f"inner bal {scores[-1]:.2f}  {cfg}")

        best = int(np.argmax(scores))
        yhat = fit_once(prop, y, tr, te, grid[best])
        bal = balanced_accuracy(y[te], yhat) * 100
        oof[str(fi)] = {"test_idx": te.tolist(), "y_pred": yhat.tolist()}
        per_fold.append({"fold": fi, "chosen": grid[best],
                         "inner_best": scores[best],
                         "inner_all": scores, "outer_bal": bal,
                         "minutes": (time.time() - t0) / 60})
        log(f"outer fold {fi + 1}/{n_outer}: chose {grid[best]}  "
            f"inner {scores[best]:.2f}  ->  OUTER {bal:.2f}  "
            f"({(time.time() - t0) / 60:.1f} min)")
        save(state)
    return state


def permutation_null(y, y_true_idx, y_pred, n=2000, seed=SEED):
    """Is the pooled prediction vector informative about the labels?

    The labels are shuffled against the FIXED out-of-fold predictions. This
    tests the prediction vector, not the whole pipeline - a full pipeline
    permutation would mean re-running the nested search per shuffle, which is
    not affordable here. Stated precisely so it is not over-read.
    """
    from metrics_fixed import balanced_accuracy
    rng = np.random.RandomState(seed)
    yt = np.asarray(y)[y_true_idx]
    obs = balanced_accuracy(yt, y_pred) * 100
    null = []
    for _ in range(n):
        null.append(balanced_accuracy(rng.permutation(yt), y_pred) * 100)
    null = np.array(null)
    return {"observed": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float((np.sum(null >= obs) + 1) / (n + 1)),
            "n_shuffles": n,
            "note": "labels shuffled against fixed OOF predictions; tests the "
                    "prediction vector, not the full pipeline"}


def save(state):
    OUT.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--budget", type=int, default=12,
                    help="configurations to sample; 0 = full grid")
    ap.add_argument("--ablate", action="store_true",
                    help="also vary opt (which published attention blocks are "
                         "built) - an architecture ablation, not tuning")
    ap.add_argument("--probe", action="store_true",
                    help="time a single fit and exit")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from optimize_models import load_features, subfunctions_lite
    subfunctions_lite()
    data = load_features()
    y = np.asarray(data["labels"]).astype(int)
    prop = np.asarray(data["proposed"], dtype=np.float32)
    log(f"proposed tensor {prop.shape}  labels {np.bincount(y).tolist()}")

    if args.probe:
        from sklearn.model_selection import StratifiedKFold
        from metrics_fixed import balanced_accuracy
        tr, te = next(StratifiedKFold(5, shuffle=True,
                                      random_state=SEED).split(prop, y))
        t0 = time.time()
        yhat = fit_once(prop, y, tr, te, OPT_RECIPE)
        dt = time.time() - t0
        log(f"one fit at {OPT_RECIPE['epochs']} epochs, batch "
            f"{OPT_RECIPE['batch_size']}, {len(tr)} train: {dt:.1f} s")
        log(f"  bal {balanced_accuracy(y[te], yhat) * 100:.2f}  "
            f"predicted classes {np.bincount(yhat, minlength=2).tolist()}")
        log(f"  a {args.outer}x{args.inner} nested search over N configs "
            f"costs about N x {args.outer * args.inner * dt / 60:.1f} min "
            f"+ {args.outer * dt / 60:.1f} min of refits")
        return

    grid = build_space(args.ablate, args.budget)
    log(f"{len(grid)} configurations, outer {args.outer} x inner {args.inner} "
        f"= {len(grid) * args.outer * args.inner + args.outer} fits")

    state = {}
    if args.resume and OUT.exists():
        state = json.loads(OUT.read_text(encoding="utf-8"))
        log(f"resumed: {len(state.get('oof', {}))} outer folds already done")
    state["grid"] = grid
    state["protocol"] = {
        "outer": args.outer, "inner": args.inner, "seed": SEED,
        "selection": "inside training folds only",
        "architecture": "published SMA-CLMPNet, unchanged"
        + (" except opt (ablation)" if args.ablate else ""),
        "scored_by": "Optimized/metrics_fixed.py",
    }
    save(state)

    t0 = time.time()
    state = nested_search(prop, y, grid, args.outer, args.inner, state)

    from metrics_fixed import balanced_accuracy
    idx = np.concatenate([np.array(v["test_idx"])
                          for _, v in sorted(state["oof"].items(),
                                             key=lambda kv: int(kv[0]))])
    pred = np.concatenate([np.array(v["y_pred"])
                           for _, v in sorted(state["oof"].items(),
                                              key=lambda kv: int(kv[0]))])
    nested = balanced_accuracy(y[idx], pred) * 100
    inner_best = float(np.mean([f["inner_best"] for f in state["per_fold"]]))

    state["result"] = {
        "nested_balanced": nested,
        "inner_best_mean": inner_best,
        "selection_bias": inner_best - nested,
        "predicted_class_counts": np.bincount(pred, minlength=2).tolist(),
        "single_class_collapse": bool(len(np.unique(pred)) == 1),
        "reference_l1_logreg_temporal_delta": 77.17,
        "beats_reference": bool(nested > 77.17),
        "minutes": (time.time() - t0) / 60,
    }
    state["permutation"] = permutation_null(y, idx, pred)
    save(state)

    log("")
    log(f"nested balanced accuracy   {nested:.2f}%   <- the honest number")
    log(f"inner-best mean            {inner_best:.2f}%   (optimistic)")
    log(f"selection bias             {inner_best - nested:.2f} points")
    log(f"permutation p              {state['permutation']['p_value']:.4f} "
        f"(null 95th {state['permutation']['null_p95']:.2f}%)")
    log(f"predicted class counts     "
        f"{state['result']['predicted_class_counts']}")
    if state["result"]["single_class_collapse"]:
        log("  COLLAPSED: every sample predicted one class - learned nothing")
    log(f"vs L1 logreg on temporal deltas (77.17%): "
        f"{'BEATS' if nested > 77.17 else 'does not beat'}")
    log(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
