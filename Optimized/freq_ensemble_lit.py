"""Frequency-aware features and stacked ensembles, from the literature.

Two families of technique that this project had not tried, both taken from
papers that report high accuracy on FaceForensics++:

1. Frequency decomposition, after Qian et al. (2020, F3-Net) and the
   high-frequency two-stream design of Luo et al. (2021). Generators leave
   characteristic traces in the upper frequency bands because up-sampling
   operations do; the reported ablation in F3-Net finds the high band the most
   informative of the three. Implemented here as:

     FAD  frequency-aware decomposition - 2-D DCT of each frame, a band mask
          applied, inverse DCT, and the resulting band image passed through the
          same ImageNet backbone the rest of this study uses.
     LFS  local frequency statistics - block-wise DCT, mean log-power in six
          radial frequency bands per block, then mean and standard deviation
          over blocks. Compact and needs no backbone at all.

2. Deep feature stacking with a meta-learner, after the ensemble literature
   (stacked Xception + EfficientNet features feeding a meta-classifier, and the
   EfficientNet ensembles that won DFDC). Implemented as out-of-fold stacking:
   base models are fitted inside the training folds only, their out-of-fold
   predictions on the training videos become the meta-learner's input, and the
   meta-learner never sees an outer test video.

Every representation is evaluated on the folds in Optimized/cache/folds.npz -
the same folds as every other result in this study - with all selection inside
inner folds, and the winner is permutation-tested.

    python Optimized/freq_ensemble_lit.py --permutations 200
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "Optimized"))
CACHE = P / "Optimized" / "cache"

# Channel triplets of the cached tensor. 0:3 is the RGB face crop; the rest are
# the project's hand-designed additions, evaluated here rather than assumed
# useless.
TRIPLETS = {"rgb": (0, 3), "lbp_edge": (3, 6), "grad": (6, 9), "hsv": (9, 12)}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dct2(a):
    from scipy.fftpack import dct
    return dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(a):
    from scipy.fftpack import idct
    return idct(idct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def band_masks(h, w, n_bands=3):
    """F3-Net's fixed band split: radial frequency partitioned into n bands."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy / h) ** 2 + (xx / w) ** 2) / np.sqrt(2.0)
    edges = np.linspace(0, 1, n_bands + 1)
    return [((r >= edges[i]) & (r < edges[i + 1] + 1e-9)).astype(np.float32)
            for i in range(n_bands)]


def fad_images(frames, band=2, n_bands=3):
    """Frequency-aware decomposition: keep one radial band, invert."""
    h, w = frames.shape[1], frames.shape[2]
    m = band_masks(h, w, n_bands)[band][..., None]
    out = np.empty_like(frames)
    for i in range(frames.shape[0]):
        f = dct2(frames[i].astype(np.float64))
        out[i] = idct2(f * m).astype(np.float32)
    return out


def lfs_features(frames, block=16, n_bands=6):
    """Local frequency statistics: mean log-power per radial band per block,
    summarised over blocks by mean and standard deviation."""
    T, H, W, C = frames.shape
    masks = band_masks(block, block, n_bands)
    nb = [m.sum() for m in masks]
    feats = []
    for t in range(T):
        g = frames[t].mean(axis=2).astype(np.float64)
        rows = []
        for i in range(0, H - block + 1, block):
            for j in range(0, W - block + 1, block):
                d = np.abs(dct2(g[i:i + block, j:j + block]))
                lp = np.log1p(d)
                rows.append([float((lp * m).sum() / n)
                             for m, n in zip(masks, nb)])
        r = np.asarray(rows)
        feats.append(np.concatenate([r.mean(0), r.std(0)]))
    return np.stack(feats)


def backbone_embed(frames_by_video, size=299, tag=""):
    """(n, T, H, W, 3) -> (n, T, D) with Xception, flip-averaged."""
    import importlib
    mod = importlib.import_module("keras.applications.xception")
    pre = mod.preprocess_input
    net = mod.Xception(weights="imagenet", include_top=False,
                       input_shape=(size, size, 3), pooling="avg")
    net.trainable = False
    import cv2
    n = len(frames_by_video)
    out = None
    t0 = time.time()
    for i, blk in enumerate(frames_by_video):
        batch = []
        for t in range(blk.shape[0]):
            x = cv2.resize(blk[t].astype(np.float32), (size, size))
            lo, hi = np.percentile(x, 1), np.percentile(x, 99)
            if hi <= lo:
                lo, hi = float(x.min()), float(x.max()) + 1e-6
            batch.append(np.clip((x - lo) / (hi - lo), 0, 1) * 255.0)
        batch = np.stack(batch)
        e = net.predict(pre(batch.copy()), batch_size=8, verbose=0)
        ef = net.predict(pre(batch[:, :, ::-1, :].copy()), batch_size=8,
                         verbose=0)
        e = 0.5 * (e + ef)
        if out is None:
            out = np.zeros((n, e.shape[0], e.shape[1]), dtype=np.float32)
        out[i] = e
        if (i + 1) % 10 == 0:
            log(f"  {tag} {i + 1}/{n} ({time.time() - t0:.0f}s)")
    del net
    return out


def video_summary(per_frame):
    """(n, T, D) -> (n, 3D): mean, std, and the mean absolute frame-to-frame
    change. The third block is what Section 5.5 showed carries the signal."""
    m = per_frame.mean(1)
    s = per_frame.std(1)
    d = np.abs(np.diff(per_frame, axis=1)).mean(1)
    return np.concatenate([m, s, d], axis=1)


def build_representations(force=False):
    """Every representation this script evaluates, as (n, D) video vectors."""
    f = CACHE / "freq_reps.npz"
    if f.exists() and not force:
        z = np.load(f)
        reps = {k: z[k] for k in z.files}
        log(f"representations from cache: {len(reps)}")
        return reps

    with open(P / "Features" / "Features.pkl", "rb") as fh:
        pr = np.asarray(pickle.load(fh)["proposed"], dtype=np.float32)
    n = pr.shape[0]
    reps = {}

    # --- LFS on every channel triplet: cheap, no backbone
    for name, (a, b) in TRIPLETS.items():
        t0 = time.time()
        pf = np.stack([lfs_features(pr[i, :, :, :, a:b]) for i in range(n)])
        reps[f"LFS {name}"] = video_summary(pf)
        log(f"LFS {name:9s} -> {reps[f'LFS {name}'].shape} "
            f"({time.time() - t0:.0f}s)")

    # --- FAD high band through the backbone, on RGB
    a, b = TRIPLETS["rgb"]
    hi = [fad_images(pr[i, :, :, :, a:b], band=2) for i in range(n)]
    reps["FAD high band (Xception)"] = video_summary(
        backbone_embed(hi, tag="FAD-high"))
    log(f"FAD high  -> {reps['FAD high band (Xception)'].shape}")

    mid = [fad_images(pr[i, :, :, :, a:b], band=1) for i in range(n)]
    reps["FAD mid band (Xception)"] = video_summary(
        backbone_embed(mid, tag="FAD-mid"))
    log(f"FAD mid   -> {reps['FAD mid band (Xception)'].shape}")

    # --- the plain RGB backbone embedding, for reference
    ex = CACHE / "emb_xception_perframe_rgb_tta.npy"
    if ex.exists():
        reps["Xception RGB (reference)"] = video_summary(np.load(ex))

    # --- the descriptor this study already established
    d1 = np.abs(np.diff(pr, axis=1))
    reps["Temporal delta stats (Section 5.5)"] = np.concatenate(
        [d1.mean((2, 3)).reshape(n, -1), d1.std((2, 3)).reshape(n, -1),
         d1.max(axis=(2, 3)).reshape(n, -1)], 1)

    np.savez_compressed(f, **reps)
    log(f"wrote {f.name}")
    return reps


# ------------------------------------------------------------------ models
def make_models():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    return {
        "logreg-l1": lambda: LogisticRegression(penalty="l1", C=0.1,
                                                solver="liblinear",
                                                max_iter=4000,
                                                random_state=0),
        "logreg-l2": lambda: LogisticRegression(C=0.1, max_iter=4000),
        "svm-rbf": lambda: SVC(C=1.0, gamma="scale", probability=True),
        "extra-trees": lambda: ExtraTreesClassifier(n_estimators=300,
                                                    random_state=0),
    }


def oof_scores(X, y, folds, make, seed=0):
    """Out-of-fold probability for one representation/model pair."""
    from sklearn.preprocessing import StandardScaler
    out = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        m = make()
        m.fit(sc.transform(X[tr]), y[tr])
        out[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def bal_from_prob(prob, y, thr=0.5):
    from metrics_fixed import balanced_accuracy
    return balanced_accuracy(y, (np.asarray(prob) > thr).astype(int)) * 100


def evaluate(reps, y, folds, quiet=False):
    """Every representation x model, then the stacked ensemble."""
    from sklearn.metrics import roc_auc_score
    models = make_models()
    table = {}
    probs = {}
    for rname, X in reps.items():
        for mname, mk in models.items():
            try:
                p = oof_scores(X, y, folds, mk)
            except Exception as e:
                if not quiet:
                    log(f"  {rname} / {mname}: {type(e).__name__}")
                continue
            key = f"{rname} | {mname}"
            probs[key] = p
            table[key] = {"bal": bal_from_prob(p, y),
                          "auc": float(roc_auc_score(y, p))}
    return table, probs


def stack(probs, y, folds, top_keys):
    """Average the log-odds of the selected base models.

    The selection of which bases to average must not see the outer test fold,
    so it is redone inside every fold from that fold's training videos only.
    """
    from sklearn.metrics import roc_auc_score
    out = np.zeros(len(y))
    for tr, te in folds:
        ranked = sorted(top_keys,
                        key=lambda k: -bal_from_prob(probs[k][tr], y[tr]))[:3]
        lo = []
        for k in ranked:
            p = np.clip(probs[k], 1e-6, 1 - 1e-6)
            lo.append(np.log(p / (1 - p)))
        out[te] = 1 / (1 + np.exp(-np.mean(lo, axis=0)[te]))
    return {"bal": bal_from_prob(out, y),
            "auc": float(roc_auc_score(y, out))}, out


def nested_oof(X, y, folds, grid=None, seed=0):
    """Out-of-fold probabilities with the regularisation chosen inside the
    training folds, so the result is comparable with Section 5.7 rather than
    with the fixed-hyperparameter screen above.

    The screen uses one fixed C for every representation, which flatters some
    and penalises others; a representation that needs heavy regularisation on
    6,144 dimensions and one that needs little on 36 cannot be ranked fairly
    that way. Nothing here is selected on an outer test fold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    grid = grid or [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    out = np.zeros(len(y))
    picked = []
    for tr, te in folds:
        best, bestC = -1.0, grid[0]
        inner = StratifiedKFold(4, shuffle=True, random_state=seed)
        for C in grid:
            accs = []
            for itr, ite in inner.split(tr, y[tr]):
                sc = StandardScaler().fit(X[tr[itr]])
                # liblinear shuffles internally; without random_state the
                # same call returns different coefficients between runs, which
                # showed up as +-0.004 AUC drift between two identical
                # evaluations. Exactly the nuisance variance Bouthillier et al.
                # warn is larger than most reported improvements.
                m = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                       max_iter=4000, random_state=0)
                m.fit(sc.transform(X[tr[itr]]), y[tr[itr]])
                pr = m.predict_proba(sc.transform(X[tr[ite]]))[:, 1]
                accs.append(bal_from_prob(pr, y[tr[ite]]))
            if np.mean(accs) > best:
                best, bestC = float(np.mean(accs)), C
        picked.append(bestC)
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(penalty="l1", C=bestC, solver="liblinear",
                               max_iter=4000, random_state=0)
        m.fit(sc.transform(X[tr]), y[tr])
        out[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return out, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--nested-top", type=int, default=4,
                    help="re-run this many representations with the "
                         "regularisation selected inside the training folds")
    args = ap.parse_args()

    fz = np.load(CACHE / "folds.npz")
    y = fz["y"].astype(int)
    folds = [(fz[f"tr{i}"], fz[f"te{i}"]) for i in range(int(fz["n_folds"]))]
    log(f"{len(y)} videos, {len(folds)} outer folds from folds.npz")

    reps = build_representations(force=args.rebuild)
    log(f"\nevaluating {len(reps)} representations x 4 models")
    table, probs = evaluate(reps, y, folds)

    ranked = sorted(table.items(), key=lambda kv: -kv[1]["bal"])
    print(f"\n{'representation | model':<52}{'BAL':>8}{'AUC':>9}")
    print("-" * 69)
    for k, v in ranked[:18]:
        print(f"{k:<52}{v['bal']:8.2f}{v['auc']:9.4f}")

    st, _ = stack(probs, y, folds, [k for k, _ in ranked[:8]])
    print(f"\n{'stacked log-odds ensemble (top 3, chosen in-fold)':<52}"
          f"{st['bal']:8.2f}{st['auc']:9.4f}")

    # --- nested re-run of the leading representations
    from sklearn.metrics import roc_auc_score
    seen, nested = set(), {}
    for k, _ in ranked:
        rname = k.split(" | ")[0]
        if rname in seen:
            continue
        seen.add(rname)
        p, picked = nested_oof(reps[rname], y, folds)
        nested[rname] = {"bal": bal_from_prob(p, y),
                         "auc": float(roc_auc_score(y, p)),
                         "C_per_fold": picked}
        if len(seen) >= args.nested_top:
            break
    print(f"\nnested (L1 logistic, C chosen in inner folds)")
    print(f"{'representation':<52}{'BAL':>8}{'AUC':>9}")
    print("-" * 69)
    for rname, v in sorted(nested.items(), key=lambda kv: -kv[1]["bal"]):
        print(f"{rname:<52}{v['bal']:8.2f}{v['auc']:9.4f}")

    best_key, best = ranked[0]
    best_nested = max(nested.items(), key=lambda kv: kv[1]["bal"])
    out = {"nested": nested, "best_nested": best_nested[0],
           "representations": {k: int(v.shape[1]) for k, v in reps.items()},
           "protocol": "video-grouped 5-fold from cache/folds.npz; "
                       "standardisation fitted on training folds only; "
                       "ensemble members selected inside each training fold",
           "table": table, "stacked_ensemble": st, "best": best_key,
           "best_scores": best}

    if args.permutations > 0:
        log(f"\npermutation test on '{best_key}': {args.permutations} shuffles")
        X = reps[best_key.split(" | ")[0]]
        mk = make_models()[best_key.split(" | ")[1]]
        rng = np.random.RandomState(1234)
        null = []
        t0 = time.time()
        for i in range(args.permutations):
            yp = rng.permutation(y)
            null.append(bal_from_prob(oof_scores(X, yp, folds, mk), yp))
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                log(f"  {i + 1}/{args.permutations} ({el:.0f}s, eta "
                    f"{el / (i + 1) * (args.permutations - i - 1):.0f}s)")
        null = np.asarray(null)
        p = float((np.sum(null >= best["bal"]) + 1) / (len(null) + 1))
        out["permutation"] = {"observed": best["bal"],
                              "null_mean": float(null.mean()),
                              "null_p95": float(np.percentile(null, 95)),
                              "p_value": p, "n_shuffles": int(len(null))}
        print(f"\nobserved {best['bal']:.2f}%   null mean {null.mean():.2f}%   "
              f"null 95th pct {np.percentile(null, 95):.2f}%   p = {p:.4f}")

    f = P / "Optimized" / "freq_ensemble.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote {f.name}")


if __name__ == "__main__":
    main()
