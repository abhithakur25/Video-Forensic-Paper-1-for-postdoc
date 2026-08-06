"""Out-of-fold ROC, AUC and confusion matrix for the best honest pipeline.

Sections 5.9 and 5.10 of the paper need a confusion matrix and an ROC curve.
The evaluation harness stores hard predictions only, so neither could be
reported from it. This recomputes both from scratch for the winning pipeline
identified in optimize_v2.py -- L1 logistic regression on frame-to-frame
temporal delta statistics -- and for the proposed SMA-CLMPNet feature summary
as a reference point.

Protocol is the same nested cross-validation as optimize_v2.py: hyper-
parameters are chosen by GridSearchCV on 4 inner folds inside each of 5 outer
folds, and the score written out is the outer-fold prediction, which the
selection never saw. Probabilities are collected out-of-fold across all 50
samples, so the ROC is over a full held-out pass rather than a single split.

The published pipeline's own ROC cannot be reproduced honestly: Analysis.py
builds it through Evaluation_Metrics1, i.e. the tampered scorer, so its curve
is a function of a random vector rather than of any model output.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
SEED = 1234


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def temporal_delta_stats(pr):
    """The winning representation, verbatim from optimize_v2.py:101-106."""
    n = pr.shape[0]
    d1 = np.abs(np.diff(pr, axis=1))
    return np.concatenate([d1.mean((2, 3)).reshape(n, -1),
                           d1.std((2, 3)).reshape(n, -1),
                           d1.max(axis=(2, 3)).reshape(n, -1)], 1)


def per_frame_mean_std(pr):
    n = pr.shape[0]
    return np.concatenate([pr.mean((2, 3)).reshape(n, -1),
                           pr.std((2, 3)).reshape(n, -1)], 1)


def oof_scores(X, y):
    """Out-of-fold probabilities under the same nested CV as optimize_v2."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("sc", StandardScaler()), ("sel", "passthrough"),
                     ("clf", LogisticRegression(max_iter=5000, penalty="l1",
                                                solver="liblinear",
                                                class_weight="balanced"))])
    grid = {"sel": ["passthrough",
                    SelectKBest(f_classif, k=10), SelectKBest(f_classif, k=25),
                    PCA(n_components=8, random_state=SEED),
                    PCA(n_components=20, random_state=SEED)],
            "clf__C": [0.01, 0.1, 1, 10]}

    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(4, shuffle=True, random_state=SEED)
    prob = np.full(len(y), np.nan)
    pred = np.full(len(y), -1, int)
    for tr, te in outer.split(X, y):
        gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=inner,
                          n_jobs=1, error_score="raise")
        gs.fit(X[tr], y[tr])
        prob[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
        pred[te] = gs.best_estimator_.predict(X[te])
    assert not np.isnan(prob).any() and (pred >= 0).all()
    return prob, pred


def main():
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 roc_auc_score, roc_curve)

    with open(P / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    y = np.asarray(data["labels"]).astype(int)
    pr = np.asarray(data["proposed"], dtype=np.float32)
    log(f"{len(y)} videos, {int((y == 0).sum())} authentic / "
        f"{int((y == 1).sum())} forged; proposed tensor {pr.shape}")

    out = {"corpus": {"n": int(len(y)), "authentic": int((y == 0).sum()),
                      "forged": int((y == 1).sum())},
           "protocol": "nested CV, outer 5-fold, inner 4-fold, "
                       "out-of-fold probabilities over all samples",
           "seed": SEED, "curves": {}}

    for name, X in [("temporal delta stats (best honest pipeline)",
                     temporal_delta_stats(pr)),
                    ("per-frame mean+std (time-collapsed reference)",
                     per_frame_mean_std(pr))]:
        log(f"{name}: X{X.shape}")
        prob, pred = oof_scores(X, y)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y, prob)
        auc = float(roc_auc_score(y, prob))
        bal = float(balanced_accuracy_score(y, pred))
        tn, fp, fn, tp = cm.ravel()
        log(f"    AUC {auc:.4f}  bal-acc {bal*100:.2f}  "
            f"cm [[{tn} {fp}] [{fn} {tp}]]")
        out["curves"][name] = {
            "auc": auc, "balanced_accuracy": bal,
            "confusion_matrix": {"labels": ["authentic(0)", "forged(1)"],
                                 "TN": int(tn), "FP": int(fp),
                                 "FN": int(fn), "TP": int(tp)},
            "accuracy": float((tn + tp) / cm.sum()),
            "sensitivity_forged": float(tp / max(1, tp + fn)),
            "specificity_authentic": float(tn / max(1, tn + fp)),
            "fpr": [round(float(v), 6) for v in fpr],
            "tpr": [round(float(v), 6) for v in tpr],
        }

    # AUC null: is the curve distinguishable from label-shuffled chance?
    log("permutation null on AUC, 200 shuffles")
    X = temporal_delta_stats(pr)
    rng = np.random.default_rng(SEED)
    null = []
    for i in range(200):
        yp = rng.permutation(y)
        try:
            null.append(float(roc_auc_score(yp, oof_scores(X, yp)[0])))
        except ValueError:
            continue
        if (i + 1) % 50 == 0:
            log(f"    {i+1}/200")
    obs = out["curves"]["temporal delta stats (best honest pipeline)"]["auc"]
    null = np.asarray(null)
    out["auc_permutation"] = {
        "observed": obs, "n_shuffles": int(len(null)),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float((np.sum(null >= obs) + 1) / (len(null) + 1)),
    }
    log(f"AUC {obs:.4f}  null mean {null.mean():.4f}  "
        f"p95 {np.percentile(null, 95):.4f}  "
        f"p = {out['auc_permutation']['p_value']:.4f}")

    f = P / "Optimized" / "roc_confusion.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote {f.relative_to(P)}")


if __name__ == "__main__":
    main()
