"""Serious attempt to extract signal: richer representations + model selection.

The first pass showed every deep model collapsing onto one class and a
channel-mean probe sitting at chance. Neither is a fair test:

  * the deep models trained for 10 epochs at batch 32 on 19-44 samples, i.e.
    ~20 weight updates, on unnormalised inputs spanning -24.8 to 255;
  * the probe reduced a (10, 128, 128, 12) tensor to 24 numbers.

This module builds representations that actually preserve the structure a
forgery detector would use - spatial layout at several scales, per-channel
distributions, and frame-to-frame temporal change, which is where splicing and
face-swap artefacts live - and then does honest model selection over them.

METHODOLOGY
-----------
Nested cross-validation throughout. With 50 samples, choosing the best of N
pipelines on cross-validated score and then reporting that score is
optimistically biased - the selection has seen every fold. The inner loop
selects hyper-parameters, the outer loop estimates performance on data the
selection never touched. The outer score is the one reported.

Scored by balanced accuracy, because the corpus is 29/21 and plain accuracy
rewards a constant predictor with 58%.

The best representation is then permutation-tested: if the honest score cannot
beat the 95th percentile of scores obtained on shuffled labels, there is no
signal, and no amount of further modelling will create one.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

P = Path(__file__).resolve().parents[1]
SEED = 1234
N_JOBS = 6      # 20 cores available, TF sweep is using some
rng = np.random.default_rng(SEED)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------ representations
def grid_pool(x, grids=(1, 2, 4)):
    """Multi-scale spatial average pooling: keeps coarse spatial layout that a
    global mean destroys. x is (H, W, C) -> concatenated grid cell means."""
    H, W, C = x.shape
    out = []
    for g in grids:
        hs, ws = H // g, W // g
        for i in range(g):
            for j in range(g):
                out.append(x[i*hs:(i+1)*hs, j*ws:(j+1)*ws].mean((0, 1)))
    return np.concatenate(out)


def chan_hist(x, bins=8):
    """Per-channel intensity histogram - distribution shape, not just its mean."""
    C = x.shape[-1]
    v = x.reshape(-1, C)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    if not np.isfinite(lo) or hi <= lo:
        lo, hi = float(v.min()), float(v.max()) + 1e-6
    return np.concatenate([np.histogram(v[:, c], bins=bins, range=(lo, hi),
                                        density=True)[0] for c in range(C)])


def build(data):
    lab = np.asarray(data["labels"]).astype(int)
    c1 = np.asarray(data["comparative1"], dtype=np.float32)   # (50,128,128,10)
    c4 = np.asarray(data["comparative4"], dtype=np.float32)   # (50,10,12)
    pr = np.asarray(data["proposed"], dtype=np.float32)       # (50,10,128,128,12)
    n = len(lab)
    reps = {}

    reps["c1: multiscale grid pool"] = np.stack([grid_pool(c1[i])
                                                 for i in range(n)])
    reps["c1: channel histograms"] = np.stack([chan_hist(c1[i])
                                               for i in range(n)])
    reps["c1: grid pool + hist"] = np.concatenate(
        [reps["c1: multiscale grid pool"], reps["c1: channel histograms"]], 1)

    reps["c4: GLCM stats (flat)"] = c4.reshape(n, -1)

    # proposed: spatial summary per frame per channel, keeping the time axis
    sp_mean = pr.mean((2, 3))                                  # (50,10,12)
    sp_std = pr.std((2, 3))                                    # (50,10,12)
    reps["proposed: per-frame mean+std"] = np.concatenate(
        [sp_mean.reshape(n, -1), sp_std.reshape(n, -1)], 1)

    # temporal change: face-swap and splice artefacts show up as frame-to-frame
    # inconsistency, which any time-collapsed summary erases entirely
    d1 = np.abs(np.diff(pr, axis=1))                           # (50,9,H,W,12)
    reps["proposed: temporal delta stats"] = np.concatenate([
        d1.mean((2, 3)).reshape(n, -1),
        d1.std((2, 3)).reshape(n, -1),
        d1.max(axis=(2, 3)).reshape(n, -1),
    ], 1)

    reps["proposed: frame0 grid pool"] = np.stack([grid_pool(pr[i, 0])
                                                   for i in range(n)])
    reps["proposed: mean-frame grid pool"] = np.stack(
        [grid_pool(pr[i].mean(0)) for i in range(n)])

    reps["proposed: spatial + temporal"] = np.concatenate(
        [reps["proposed: per-frame mean+std"],
         reps["proposed: temporal delta stats"]], 1)

    cache = P / "Optimized" / "cache"
    for f in sorted(cache.glob("emb_*.npy")):
        reps[f"{f.stem[4:]} embedding"] = np.load(f)

    # everything that is cheap, concatenated
    reps["ALL (grid+hist+temporal+GLCM)"] = np.concatenate([
        reps["c1: grid pool + hist"], reps["c4: GLCM stats (flat)"],
        reps["proposed: spatial + temporal"]], 1)
    return reps, lab


# --------------------------------------------------------------- model search
def pipelines():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def pipe(clf):
        return Pipeline([("sc", StandardScaler()),
                         ("sel", "passthrough"),
                         ("clf", clf)])

    reducers = ["passthrough",
                SelectKBest(f_classif, k=10), SelectKBest(f_classif, k=25),
                PCA(n_components=8, random_state=SEED),
                PCA(n_components=20, random_state=SEED)]

    return {
        "logreg-l2": (pipe(LogisticRegression(max_iter=5000,
                                              class_weight="balanced")),
                      {"sel": reducers, "clf__C": [0.01, 0.1, 1, 10]}),
        "logreg-l1": (pipe(LogisticRegression(max_iter=5000, penalty="l1",
                                              solver="liblinear",
                                              class_weight="balanced")),
                      {"sel": reducers, "clf__C": [0.01, 0.1, 1, 10]}),
        "svm-rbf": (pipe(SVC(class_weight="balanced")),
                    {"sel": reducers, "clf__C": [0.1, 1, 10],
                     "clf__gamma": ["scale", 0.01, 0.001]}),
        "svm-lin": (pipe(SVC(kernel="linear", class_weight="balanced")),
                    {"sel": reducers, "clf__C": [0.01, 0.1, 1, 10]}),
        "rf": (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED,
                                           class_weight="balanced")),
               {"sel": reducers, "clf__max_depth": [None, 3, 5],
                "clf__min_samples_leaf": [1, 3]}),
        "extra-trees": (pipe(ExtraTreesClassifier(n_estimators=300,
                                                  random_state=SEED,
                                                  class_weight="balanced")),
                        {"sel": reducers, "clf__max_depth": [None, 3, 5]}),
        "grad-boost": (pipe(GradientBoostingClassifier(random_state=SEED)),
                       {"sel": reducers, "clf__n_estimators": [50, 150],
                        "clf__max_depth": [1, 2]}),
        "knn": (pipe(KNeighborsClassifier()),
                {"sel": reducers, "clf__n_neighbors": [3, 5, 9],
                 "clf__weights": ["uniform", "distance"]}),
        "gaussian-nb": (pipe(GaussianNB()), {"sel": reducers}),
    }


def nested_score(X, y, est, grid, n_outer=5, n_inner=4, seed=SEED):
    """Unbiased estimate: hyper-parameters chosen inside each outer fold."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
    inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    # The inner grid is the bulk of the work and parallelises cleanly; the
    # outer loop stays serial so the two levels do not fight for cores.
    search = GridSearchCV(est, grid, scoring="balanced_accuracy", cv=inner,
                          n_jobs=N_JOBS, refit=True)
    s = cross_val_score(search, X, y, cv=outer, scoring="balanced_accuracy",
                        n_jobs=1)
    return float(s.mean()), float(s.std())


def main():
    import argparse
    import pickle

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; evaluate only "
                         "representations whose name contains one of them")
    ap.add_argument("--perms", type=int, default=100,
                    help="permutations for the null; 0 skips the test")
    ap.add_argument("--out", default="optimize_v2.json")
    args = ap.parse_args()

    with open(P / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    reps, y = build(data)
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        reps = {r: X for r, X in reps.items()
                if any(k.lower() in r.lower() for k in keys)}
        if not reps:
            raise SystemExit(f"--only {args.only!r} matched no representation")
    log(f"{len(reps)} representations, {len(y)} samples, "
        f"classes {np.bincount(y).tolist()}")

    pipes = pipelines()
    table = {}
    for rname, X in reps.items():
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        row = {}
        for mname, (est, grid) in pipes.items():
            m, s = nested_score(X, y, est, grid)
            row[mname] = (m, s)
        best = max(row, key=lambda k: row[k][0])
        table[rname] = row
        log(f"{rname:<34} dim={X.shape[1]:<5} best={best} "
            f"{row[best][0]*100:.2f}% ±{row[best][1]*100:.2f}")

    # ------------------------------------------------------------ the winner
    flat = [(r, m, v[0], v[1]) for r, row in table.items()
            for m, v in row.items()]
    flat.sort(key=lambda t: -t[2])
    log("")
    log("top 10 (representation, model, nested balanced accuracy):")
    for r, m, v, s in flat[:10]:
        log(f"   {v*100:6.2f}% ±{s*100:5.2f}   {m:<12} on {r}")

    br, bm, bv, bs = flat[0]
    log("")
    if args.perms == 0:
        log("permutation test skipped (--perms 0)")
        return
    log(f"permutation test on the winner: {bm} / {br}  ({args.perms} shuffles)")
    est, grid = pipes[bm]
    X = np.nan_to_num(np.asarray(reps[br], dtype=np.float64))
    null = []
    for i in range(args.perms):
        yp = rng.permutation(y)
        null.append(nested_score(X, yp, est, grid, seed=SEED + i)[0])
    null = np.asarray(null)
    p = float((np.sum(null >= bv) + 1) / (len(null) + 1))
    log(f"   observed {bv*100:.2f}%")
    log(f"   null mean {null.mean()*100:.2f}%  95th pct "
        f"{np.percentile(null, 95)*100:.2f}%  max {null.max()*100:.2f}%")
    log(f"   p = {p:.3f}  -> "
        f"{'SIGNAL' if p < 0.05 else 'not distinguishable from chance'}")

    (P / "Optimized" / args.out).write_text(json.dumps({
        "representations": {r: {m: list(v) for m, v in row.items()}
                            for r, row in table.items()},
        "ranking": [{"representation": r, "model": m, "mean": v, "std": s}
                    for r, m, v, s in flat],
        "winner": {"representation": br, "model": bm, "nested_bal_acc": bv,
                   "null_mean": float(null.mean()),
                   "null_p95": float(np.percentile(null, 95)),
                   "p_value": p},
        "protocol": "nested CV, outer 5-fold, inner 4-fold, balanced accuracy",
    }, indent=2), encoding="utf-8")
    log(f"wrote Optimized/{args.out}")


if __name__ == "__main__":
    main()
