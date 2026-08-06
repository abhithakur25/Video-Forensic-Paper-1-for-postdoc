"""Corpus audit: near-duplicates, separability ceiling, and attainable accuracy.

Due diligence that should precede any accuracy claim on a 50-sample corpus.
Four questions:

1. Are there near-duplicate videos? FaceForensics++ manipulated clips are named
   <target>_<source> and share footage with original <target>. If the 50
   samples contain such pairs and they straddle a split, the model recognises
   the footage rather than the manipulation, and every score is inflated.

2. Is any single feature dimension trivially separable? If one channel encodes
   the label, a linear model finds it and the problem is not a detection
   problem at all.

3. What accuracy does the measured effect size actually support? Given the
   out-of-fold AUC, the best attainable accuracy at any threshold is bounded.
   Reporting a target above that bound is reporting an impossibility.

4. How wide is the confidence interval on a 10-video test fold? This sets the
   granularity below which no two methods can be distinguished.

No model is trained here and nothing is tuned. This only characterises the
data.
"""
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]


def log(m):
    print(m, flush=True)


def main():
    from scipy.stats import beta
    from sklearn.metrics import roc_auc_score

    with open(P / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    y = np.asarray(data["labels"]).astype(int)
    pr = np.asarray(data["proposed"], dtype=np.float32)
    n = len(y)
    log(f"corpus: {n} videos, {int((y==0).sum())} authentic / "
        f"{int((y==1).sum())} forged")
    log(f"keys in Features.pkl: {sorted(data.keys())}")

    out = {"n": int(n), "authentic": int((y == 0).sum()),
           "forged": int((y == 1).sum())}

    # ---------------------------------------------------- 1. near-duplicates
    log("\n=== 1. near-duplicate check ===")
    flat = pr.reshape(n, -1).astype(np.float64)
    flat = flat - flat.mean(1, keepdims=True)
    nrm = np.linalg.norm(flat, axis=1, keepdims=True)
    C = (flat / np.maximum(nrm, 1e-12)) @ (flat / np.maximum(nrm, 1e-12)).T
    np.fill_diagonal(C, -1.0)
    pairs = [(i, j, float(C[i, j])) for i, j in combinations(range(n), 2)
             if C[i, j] > 0.98]
    pairs.sort(key=lambda t: -t[2])
    log(f"pairs with cosine > 0.98: {len(pairs)}")
    for i, j, c in pairs[:15]:
        log(f"    video {i:2d} (label {y[i]}) ~ video {j:2d} (label {y[j]})"
            f"   cos {c:.5f}"
            + ("   <-- CROSS-LABEL" if y[i] != y[j] else ""))
    out["near_duplicate_pairs_gt_098"] = len(pairs)
    out["max_offdiag_cosine"] = float(C.max())
    log(f"highest off-diagonal cosine anywhere: {C.max():.5f}")
    log("A corpus of independent videos should show no pair above ~0.98.")

    # ------------------------------------------- 2. trivially separable dim?
    log("\n=== 2. single-dimension separability ===")
    d1 = np.abs(np.diff(pr, axis=1))
    X = np.concatenate([d1.mean((2, 3)).reshape(n, -1),
                        d1.std((2, 3)).reshape(n, -1),
                        d1.max(axis=(2, 3)).reshape(n, -1)], 1)
    aucs = np.array([roc_auc_score(y, X[:, k]) for k in range(X.shape[1])])
    aucs = np.maximum(aucs, 1 - aucs)          # direction-agnostic
    order = np.argsort(-aucs)
    log(f"{X.shape[1]} temporal-delta features")
    log(f"best single-feature AUC {aucs[order[0]]:.4f} (dim {order[0]})")
    log(f"top 5: {', '.join(f'{aucs[k]:.4f}' for k in order[:5])}")
    log(f"features with AUC > 0.90: {int((aucs > 0.90).sum())}")
    log(f"features with AUC > 0.80: {int((aucs > 0.80).sum())}")
    out["best_single_feature_auc"] = float(aucs[order[0]])
    out["n_features_auc_gt_090"] = int((aucs > 0.90).sum())
    log("No dimension is trivially separable, so this is a genuine detection\n"
        "problem rather than a leaked label." if aucs[order[0]] < 0.9 else
        "A dimension is close to separable - investigate before trusting any\n"
        "downstream score.")

    # ---------------------------------------- 3. ceiling from measured AUC
    log("\n=== 3. attainable accuracy at the measured effect size ===")
    rf = P / "Optimized" / "roc_confusion.json"
    if rf.exists():
        r = json.loads(rf.read_text("utf-8"))
        c = r["curves"]["temporal delta stats (best honest pipeline)"]
        fpr = np.asarray(c["fpr"])
        tpr = np.asarray(c["tpr"])
        p1 = (y == 1).mean()
        p0 = 1 - p1
        acc = tpr * p1 + (1 - fpr) * p0
        bal = 0.5 * (tpr + (1 - fpr))
        log(f"observed out-of-fold AUC {c['auc']:.4f}")
        log(f"best accuracy over ALL thresholds on this very curve: "
            f"{acc.max()*100:.2f}%")
        log(f"best balanced accuracy over all thresholds:            "
            f"{bal.max()*100:.2f}%")
        log("Those are upper bounds obtained by picking the threshold with the")
        log("test labels in hand - i.e. already optimistic, and still nowhere")
        log("near 95%.")
        out["oof_auc"] = c["auc"]
        out["max_accuracy_any_threshold"] = float(acc.max())
        out["max_balanced_any_threshold"] = float(bal.max())
        out["null_auc_p95"] = r["auc_permutation"]["null_p95"]

    # ------------------------------------------- 4. CI on a 10-video fold
    log("\n=== 4. confidence interval on a 10-video test fold ===")
    rows = []
    for k, m in [(10, 10), (9, 10), (8, 10), (7, 10), (37, 50), (48, 50)]:
        lo = beta.ppf(0.025, k, m - k + 1) if k > 0 else 0.0
        hi = beta.ppf(0.975, k + 1, m - k) if k < m else 1.0
        rows.append((k, m, k / m, lo, hi))
        log(f"    {k:2d}/{m:2d} correct = {k/m*100:6.2f}%   "
            f"95% CI [{lo*100:.1f}, {hi*100:.1f}]")
    out["binomial_ci"] = [{"correct": k, "n": m, "acc": a, "lo": lo, "hi": hi}
                          for k, m, a, lo, hi in rows]
    log("A 10-video fold cannot distinguish 80% from 100% at 95% confidence.")

    f = P / "Optimized" / "corpus_audit.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"\nwrote {f.relative_to(P)}")


if __name__ == "__main__":
    main()
