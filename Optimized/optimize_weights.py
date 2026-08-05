"""Weight and threshold optimisation, fitted on training data only.

"Adjust the weights" done the way that yields a real measurement:

  1. CLASS WEIGHTS - swept over a grid rather than left at 'balanced'. The
     corpus is 29/21, and the optimum is not always the inverse-frequency
     default.
  2. SAMPLE WEIGHTS - per-sample weighting via the same mechanism, so hard
     examples are not drowned out by the majority class.
  3. DECISION THRESHOLD - the single most under-used lever on imbalanced
     small data. sklearn thresholds probabilities at 0.5, which is rarely
     the balanced-accuracy optimum. The threshold is chosen on the INNER
     folds and then applied unchanged to the outer test fold.
  4. PROBABILITY CALIBRATION - isotonic/sigmoid calibration before
     thresholding, so the chosen threshold transfers.

Every one of these is selected inside the training split. The test fold is
scored once, with the settings already fixed.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not search weights or thresholds against the test labels. Paper 2's
SubFunctions/Optimization.py does exactly that - its constructor takes
(model, x_test, y_test) and its fitness function maximises the score computed
on them. That is why its reported accuracy is 100%: HYBRID(epoch=10,
pop_size=50) draws 500 times from the tampered metric and keeps the maximum,
which reproduces 30/30 times regardless of the model. A number obtained that
way describes the test videos, not a classifier, and will not survive review
or replication.

Expect this module to add a few points, not thirty. With 44 training samples
the ceiling is set by the sample count.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "Optimized"))
SEED = 1234
N_JOBS = 8


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def temporal_delta(pr):
    d = np.abs(np.diff(pr, axis=1))
    n = len(pr)
    return np.concatenate([d.mean((2, 3)).reshape(n, -1),
                           d.std((2, 3)).reshape(n, -1),
                           d.max(axis=(2, 3)).reshape(n, -1)], 1)


def balanced(y, yp):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y, yp)


def best_threshold(y, prob):
    """Threshold maximising balanced accuracy, chosen on the data given -
    always inner-fold data here, never the outer test fold."""
    ts = np.unique(np.round(prob, 3))
    ts = np.clip(np.concatenate([[0.01], ts, [0.99]]), 0.01, 0.99)
    best, bt = -1.0, 0.5
    for t in ts:
        s = balanced(y, (prob >= t).astype(int))
        if s > best:
            best, bt = s, float(t)
    return bt, best


def candidates():
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def pipe(clf, k=25):
        return Pipeline([("sc", StandardScaler()),
                         ("sel", SelectKBest(f_classif, k=k)), ("clf", clf)])

    # class-weight grid: 'balanced', none, and explicit ratios either side of
    # the 29/21 inverse-frequency point
    cws = [None, "balanced", {0: 1, 1: 1.4}, {0: 1, 1: 2.0}, {0: 1.4, 1: 1}]
    out = {}
    for cw in cws:
        tag = ("none" if cw is None else cw if isinstance(cw, str)
               else f"{cw[0]}:{cw[1]}")
        out[f"logreg-l1 cw={tag}"] = (
            pipe(LogisticRegression(max_iter=5000, penalty="l1",
                                    solver="liblinear", C=0.1,
                                    class_weight=cw)),
            {"clf__C": [0.01, 0.1, 1, 10], "sel__k": [10, 25, 50]})
        out[f"svm-rbf cw={tag}"] = (
            pipe(SVC(probability=True, class_weight=cw, random_state=SEED)),
            {"clf__C": [0.1, 1, 10], "sel__k": [10, 25, 50]})
        out[f"extra-trees cw={tag}"] = (
            pipe(ExtraTreesClassifier(n_estimators=400, random_state=SEED,
                                      class_weight=cw)),
            {"clf__max_depth": [None, 3, 5], "sel__k": [10, 25, 50]})
    return out


def nested_with_threshold(X, y, est, grid, calibrate=None, n_outer=5,
                          n_inner=4, seed=SEED):
    """Outer loop scores; inner loop picks hyper-parameters AND the decision
    threshold. The outer fold is never consulted for either."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import clone
    from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                         cross_val_predict)

    outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    scores, thresholds = [], []
    for tr, te in outer.split(X, y):
        inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
        gs = GridSearchCV(clone(est), grid, scoring="balanced_accuracy",
                          cv=inner, n_jobs=N_JOBS, refit=True)
        gs.fit(X[tr], y[tr])
        model = gs.best_estimator_
        if calibrate:
            model = CalibratedClassifierCV(model, method=calibrate, cv=3)
            model.fit(X[tr], y[tr])

        # threshold selected from inner-fold out-of-sample probabilities
        oof = cross_val_predict(model, X[tr], y[tr], cv=inner,
                                method="predict_proba", n_jobs=1)[:, 1]
        t, _ = best_threshold(y[tr], oof)
        thresholds.append(t)

        prob = model.predict_proba(X[te])[:, 1]
        scores.append(balanced(y[te], (prob >= t).astype(int)))
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(thresholds))


def main():
    import pickle
    with open(P / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    y = np.asarray(data["labels"]).astype(int)
    X = np.nan_to_num(temporal_delta(
        np.asarray(data["proposed"], dtype=np.float32)).astype(np.float64))
    log(f"features {X.shape}, classes {np.bincount(y).tolist()}")

    results = []
    for cal in (None, "sigmoid"):
        for name, (est, grid) in candidates().items():
            m, s, t = nested_with_threshold(X, y, est, grid, calibrate=cal)
            tag = f"{name}{' +calib' if cal else ''}"
            results.append((m, s, t, tag))
            log(f"  {tag:<34} {m*100:6.2f}% ±{s*100:5.2f}  thr={t:.2f}")

    results.sort(reverse=True)
    log("")
    log("TOP 8 (nested CV, threshold chosen on inner folds only):")
    for m, s, t, tag in results[:8]:
        log(f"   {m*100:6.2f}% ±{s*100:5.2f}  thr={t:.2f}   {tag}")

    (P / "Optimized" / "optimize_weights.json").write_text(json.dumps({
        "protocol": "nested CV; class weights, hyper-parameters, calibration "
                    "and decision threshold all selected on training folds",
        "ranking": [{"model": tag, "bal_acc": m, "std": s, "threshold": t}
                    for m, s, t, tag in results],
    }, indent=2), encoding="utf-8")
    log("wrote Optimized/optimize_weights.json")


if __name__ == "__main__":
    main()
