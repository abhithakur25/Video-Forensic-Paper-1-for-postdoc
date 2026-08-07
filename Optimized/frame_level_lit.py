"""The literature's frame-level recipe, applied to this corpus and measured.

The published FaceForensics++ pipelines that report 95%+ do not train on video
descriptors. They train a face-cropped CNN on individual frames and aggregate
frame scores into a video decision. Rossler et al. (2019) established the
template; the aggregation step is surveyed widely, with mean softmax and mean
log-odds both standard and log-odds usually a shade better than majority vote.

That recipe has never been run in this project. Every model in Analysis1/ takes
one descriptor per video, so the effective training set is 40-45 samples. Under
the frame-level recipe it is 400-450, an order of magnitude more, at the cost of
labels that are noisier (a manipulated video's frames are not all equally
manipulated).

This script runs it. Concretely, from the literature:

  * per-frame ImageNet backbone features, face crop resized to the backbone's
    native input (Xception 299, EfficientNetV2-S 384 -> 224 here for cost)
  * frame-level classifier, video-level decision by aggregation
  * three aggregation rules compared: mean probability, mean log-odds,
    majority vote
  * horizontal-flip test-time augmentation, which costs one extra forward pass

What is deliberately kept from this project's protocol rather than the
literature's:

  * the split is by VIDEO, read from Optimized/cache/folds.npz, the same folds
    every other result in this study uses. Splitting by frame would put frames
    of one video on both sides of the boundary and inflate every number - it is
    the single easiest way to manufacture 95% here, and it is wrong.
  * regularisation strength is chosen inside inner folds, themselves grouped by
    video.
  * the observed score is compared against a null built by shuffling VIDEO
    labels and re-running the whole procedure.

    python Optimized/frame_level_lit.py --backbone xception
    python Optimized/frame_level_lit.py --backbone effv2s --permutations 200
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

# name -> (keras module, class, input size)
BACKBONES = {
    "xception": ("xception", "Xception", 299),
    "effv2s": ("efficientnet_v2", "EfficientNetV2S", 224),
    "effb0": ("efficientnet", "EfficientNetB0", 224),
}
# Channel triplets in the cached tensor. 0:3 is the RGB face crop, which is
# what the literature's detectors consume; the rest are this project's
# hand-designed additions and are evaluated separately.
RGB = (0, 3)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def to_rgb(block, size):
    """(H, W, 3) float -> backbone-ready [0, 255] at size x size.

    Same normalisation as Optimized/frame_embeddings.py, so the two are
    comparable: robust 1-99 percentile stretch, then rescale.
    """
    import cv2
    x = cv2.resize(block.astype(np.float32), (size, size))
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max()) + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1) * 255.0


def per_frame_embeddings(name, flip_tta=True, kind="rgb"):
    """(50, T, D) per-frame features, cached.

    kind='rgb'  the face crop, which is what the literature's frame-level
                detectors consume.
    kind='diff' the absolute frame-to-frame difference, which is what the
                optical-flow and temporal-coherence detectors consume
                (Amerini et al. 2019; Zheng et al. 2021; Gu et al. 2021). On
                this corpus the ROC analysis says the signal is temporal, so
                this is the variant the evidence actually points at.

    flip_tta averages the embedding with its horizontal mirror - the cheapest
    test-time augmentation and standard in every deepfake pipeline.
    """
    out = CACHE / (f"emb_{name}_perframe_{kind}"
                   f"{'_tta' if flip_tta else ''}.npy")
    if out.exists():
        e = np.load(out)
        log(f"{name}/{kind}: per-frame embeddings from cache {e.shape}")
        return e

    import importlib
    mod_name, cls_name, size = BACKBONES[name]
    with open(P / "Features" / "Features.pkl", "rb") as f:
        pr = np.asarray(pickle.load(f)["proposed"], dtype=np.float32)
    if kind == "diff":
        pr = np.abs(np.diff(pr, axis=1))          # (50, 9, H, W, 12)
    n, T = pr.shape[0], pr.shape[1]

    mod = importlib.import_module(f"keras.applications.{mod_name}")
    pre = getattr(mod, "preprocess_input", None)
    net = getattr(mod, cls_name)(weights="imagenet", include_top=False,
                                 input_shape=(size, size, 3), pooling="avg")
    net.trainable = False
    log(f"{name}: {net.count_params() / 1e6:.1f} M params, input {size}")

    feats = np.zeros((n, T, net.output_shape[-1]), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        batch = np.stack([to_rgb(pr[i, t, :, :, RGB[0]:RGB[1]], size)
                          for t in range(T)])
        b = pre(batch.copy()) if pre is not None else batch
        e = net.predict(b, batch_size=8, verbose=0)
        if flip_tta:
            bf = batch[:, :, ::-1, :]
            bf = pre(bf.copy()) if pre is not None else bf
            e = 0.5 * (e + net.predict(bf, batch_size=8, verbose=0))
        feats[i] = e
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{n} videos  ({time.time() - t0:.0f}s)")
    np.save(out, feats)
    log(f"{name}/{kind}: wrote {feats.shape} -> {out.name}")
    del net
    return feats


def aggregate(prob_frames, rule):
    """Frame probabilities (T,) for one video -> a single video probability."""
    p = np.clip(prob_frames, 1e-6, 1 - 1e-6)
    if rule == "mean_prob":
        return float(p.mean())
    if rule == "mean_logodds":
        return float(1.0 / (1.0 + np.exp(-np.log(p / (1 - p)).mean())))
    if rule == "majority":
        return float((p > 0.5).mean())
    raise ValueError(rule)


RULES = ["mean_prob", "mean_logodds", "majority"]


def run_once(emb, y, folds, rng=None, quiet=True):
    """Nested, video-grouped. Returns {rule: (video_prob, video_pred)}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n, T, D = emb.shape
    yv = y if rng is None else rng.permutation(y)
    oof = {r: np.zeros(n) for r in RULES}

    for f, (tr, te) in enumerate(folds):
        Xtr = emb[tr].reshape(-1, D)
        ytr = np.repeat(yv[tr], T)                # frame inherits video label
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)

        # C chosen inside the training videos, grouped so no video's frames
        # straddle an inner split either.
        best, bestC = -1.0, 1.0
        inner = StratifiedKFold(4, shuffle=True, random_state=0)
        for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            accs = []
            for itr, ite in inner.split(tr, yv[tr]):
                a = emb[tr[itr]].reshape(-1, D)
                b = emb[tr[ite]].reshape(-1, D)
                s2 = StandardScaler().fit(a)
                m = LogisticRegression(C=C, max_iter=2000, penalty="l2")
                m.fit(s2.transform(a), np.repeat(yv[tr[itr]], T))
                pv = m.predict_proba(s2.transform(b))[:, 1].reshape(-1, T)
                pred = np.array([aggregate(r, "mean_logodds") > 0.5
                                 for r in pv]).astype(int)
                yy = yv[tr[ite]]
                accs.append(0.5 * (((pred == 1) & (yy == 1)).sum()
                                   / max(1, (yy == 1).sum())
                                   + ((pred == 0) & (yy == 0)).sum()
                                   / max(1, (yy == 0).sum())))
            if np.mean(accs) > best:
                best, bestC = float(np.mean(accs)), C

        clf = LogisticRegression(C=bestC, max_iter=4000, penalty="l2")
        clf.fit(Xtr, ytr)
        pte = clf.predict_proba(sc.transform(emb[te].reshape(-1, D)))[:, 1]
        pte = pte.reshape(len(te), T)
        for r in RULES:
            oof[r][te] = [aggregate(row, r) for row in pte]
        if not quiet:
            log(f"  fold {f + 1}/{len(folds)}  C={bestC}  "
                f"inner bal {best * 100:.2f}%")
    return oof, yv


def score(prob, y):
    from metrics_fixed import balanced_accuracy, evaluation_metrics
    from sklearn.metrics import roc_auc_score
    pred = (np.asarray(prob) > 0.5).astype(int)
    m = evaluation_metrics(y, pred)
    return {"acc": m[0] * 100, "sen": m[1] * 100, "spe": m[2] * 100,
            "pre": (m[3] * 100 if m[3] == m[3] else float("nan")),
            "f1": m[4] * 100,
            "bal": balanced_accuracy(y, pred) * 100,
            "auc": float(roc_auc_score(y, prob))}


def leakage_demo(emb, y, seed=1234):
    """Deliberately split by FRAME instead of by video, and report what that
    does to the score.

    This is not a result and must never be quoted as one. It is here because
    it is the single easiest way to produce a 95% number on this corpus, it is
    a two-line change from the correct protocol, and the resulting figure is
    indistinguishable in a table from an honest one. Ten frames of the same
    video are near-identical; scattering them across the split boundary means
    the classifier is asked, at test time, about frames whose neighbours it
    memorised during training. It is recognising footage, not manipulation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from metrics_fixed import balanced_accuracy, evaluation_metrics

    n, T, D = emb.shape
    X = emb.reshape(-1, D)
    yf = np.repeat(y, T)
    oof = np.zeros(len(yf))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, yf):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=0.1, max_iter=4000)
        m.fit(sc.transform(X[tr]), yf[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    pred = (oof > 0.5).astype(int)
    mm = evaluation_metrics(yf, pred)
    return {"level": "frame (LEAKY - frames of one video on both sides)",
            "n_samples": int(len(yf)),
            "acc": mm[0] * 100, "sen": mm[1] * 100, "spe": mm[2] * 100,
            "f1": mm[4] * 100,
            "bal": balanced_accuracy(yf, pred) * 100,
            "auc": float(roc_auc_score(yf, oof))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="xception",
                    choices=sorted(BACKBONES))
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--leakage-demo", action="store_true",
                    help="also run the frame-level split, which leaks, to "
                         "show what it does to the number")
    ap.add_argument("--input", default="rgb", choices=["rgb", "diff", "both"],
                    help="rgb = face crop (spatial cue); diff = frame-to-frame "
                         "difference (temporal cue); both = concatenated")
    args = ap.parse_args()

    fz = np.load(CACHE / "folds.npz")
    y = fz["y"].astype(int)
    folds = [(fz[f"tr{i}"], fz[f"te{i}"]) for i in range(int(fz["n_folds"]))]
    log(f"corpus {len(y)} videos, {(y == 0).sum()} authentic / "
        f"{(y == 1).sum()} forged; {len(folds)} outer folds from folds.npz")

    tta = not args.no_tta
    if args.input == "both":
        a = per_frame_embeddings(args.backbone, tta, "rgb")
        b = per_frame_embeddings(args.backbone, tta, "diff")
        # rgb has T frames, diff has T-1; align on the shorter axis so each
        # sample pairs frame t with the difference entering frame t+1
        m = min(a.shape[1], b.shape[1])
        emb = np.concatenate([a[:, :m], b[:, :m]], axis=2)
        log(f"concatenated rgb+diff -> {emb.shape}")
    else:
        emb = per_frame_embeddings(args.backbone, tta, args.input)
    n, T, D = emb.shape
    log(f"frame-level samples: {n * T} ({n} videos x {T} frames), dim {D}")
    log("split is by VIDEO - frames never straddle a fold boundary")

    log("\nrunning the observed pipeline")
    oof, _ = run_once(emb, y, folds, quiet=False)
    results = {r: score(oof[r], y) for r in RULES}
    print()
    hdr = f"{'aggregation':<14}" + "".join(
        f"{k.upper():>9}" for k in ("acc", "sen", "spe", "pre", "f1", "bal"))
    print(hdr + f"{'AUC':>9}")
    print("-" * len(hdr + "     AUC "))
    for r in RULES:
        s = results[r]
        print(f"{r:<14}" + "".join(f"{s[k]:9.2f}" for k in
                                   ("acc", "sen", "spe", "pre", "f1", "bal"))
              + f"{s['auc']:9.4f}")

    best_rule = max(RULES, key=lambda r: results[r]["bal"])
    out = {"backbone": args.backbone, "input": args.input,
           "tta_flip": not args.no_tta,
           "frames_per_video": int(T), "embedding_dim": int(D),
           "protocol": "video-grouped 5-fold from cache/folds.npz; frame-level "
                       "L2 logistic regression; C selected in inner folds "
                       "grouped by video; video decision by aggregation",
           "results": results, "best_rule": best_rule}

    if args.leakage_demo:
        log("\nleakage demonstration - NOT a result")
        ld = leakage_demo(emb, y)
        out["leakage_demo"] = ld
        print(f"  split by frame instead of by video: "
              f"acc {ld['acc']:.2f}%  bal {ld['bal']:.2f}%  "
              f"AUC {ld['auc']:.4f}  over {ld['n_samples']} frame samples")
        print(f"  correct, video-grouped, same features:  "
              f"bal {results[best_rule]['bal']:.2f}%")
        print("  the difference is entirely the split. Nothing else changed.")

    if args.permutations > 0:
        log(f"\npermutation test: {args.permutations} shuffles of the VIDEO "
            f"labels, whole pipeline re-run each time")
        rng = np.random.RandomState(1234)
        null = []
        t0 = time.time()
        for i in range(args.permutations):
            o, yv = run_once(emb, y, folds, rng=rng)
            null.append(score(o[best_rule], yv)["bal"])
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                log(f"  {i + 1}/{args.permutations}  "
                    f"({el:.0f}s, eta {el / (i + 1) * (args.permutations - i - 1):.0f}s)")
        null = np.asarray(null)
        obs = results[best_rule]["bal"]
        p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
        out["permutation"] = {"rule": best_rule, "observed": obs,
                              "n_shuffles": int(len(null)),
                              "null_mean": float(null.mean()),
                              "null_p95": float(np.percentile(null, 95)),
                              "p_value": p}
        print(f"\n{best_rule}: observed {obs:.2f}%   null mean "
              f"{null.mean():.2f}%   null 95th pct "
              f"{np.percentile(null, 95):.2f}%   p = {p:.4f}")

    f = P / "Optimized" / f"frame_level_{args.backbone}_{args.input}.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"\nwrote {f.name}")


if __name__ == "__main__":
    main()
