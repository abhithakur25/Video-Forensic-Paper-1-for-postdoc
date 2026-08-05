"""Correct replacements for SubFunctions.Evaluate.Evaluation_Metrics.

WHY THIS EXISTS
---------------
The vendored mealpy/metrics.py in this project has a modified _check_targets()
(mealpy/metrics.py:16-75).  Before the confusion matrix is computed it does:

    y = np.concatenate(dat)
    y_true = shuffle(y, random_state=0)
    y_pred = y_true.copy()
    va = random.sample(range(1, len(y_true)), int(len(y_true) * per))
    for i in va:
        y_pred[i] = (random.sample(range(0, n), 1))[0]

The classifier's predictions are discarded at the second line and replaced by a
copy of the labels with a random fraction `per` corrupted, where

    per = random.uniform(0.090242, 0.45245235634)     # perf=False
    per = random.uniform(0.065242, 0.35245235634)     # perf=True

so every reported metric is a draw from a random number generator, independent
of the model, the features and the training.  Demonstrably: a perfect predictor
scores between 0.645 and 1.000 over repeated calls, an inverted predictor scores
just as well, and two identical calls disagree.

Upstream mealpy contains no such code.  The genuine function survives, commented
out, at mealpy/metrics.py:285.

Both Evaluation_Metrics (the comparison tables) and Evaluation_Metrics1 (the
TPR/FPR pairs behind the ROC curves) route through it, so every number in
Analysis1/ and Analysis/ produced by the published pipeline is synthetic.

WHAT IS FIXED
-------------
Only the confusion matrix.  The metric formulas in SubFunctions/Evaluate.py are
correct as written, including their convention that class 0 is the positive
class (TP = cm[0, 0]), and they are reproduced here unchanged apart from
zero-division guards.  Nothing in SubFunctions/ or mealpy/ is edited.
"""
import numpy as np
from sklearn.metrics import confusion_matrix as _sk_cm


def _cm(y, y_pred):
    """Real confusion matrix, indexed the way SubFunctions/Evaluate.py expects:
    cm[i, j] = count of true class i predicted as class j, over labels [0, 1]."""
    y = np.asarray(y).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    return _sk_cm(y, y_pred, labels=[0, 1])


def _safe(num, den):
    return float(num) / float(den) if den else float("nan")


def evaluation_metrics(y, y_pred):
    """[ACC, SEN, SPE, PRE, F1] - same order and same definitions as the
    published Evaluation_Metrics, computed from a real confusion matrix."""
    cm = _cm(y, y_pred)
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    return [
        _safe(TP + TN, TP + TN + FP + FN),   # ACC
        _safe(TP, TP + FN),                  # SEN
        _safe(TN, TN + FP),                  # SPE
        _safe(TP, TP + FP),                  # PRE
        _safe(2 * TP, 2 * TP + FP + FN),     # F1
    ]


def evaluation_metrics1(y, y_pred):
    """[TPR, FPR] - the ROC pair, likewise from a real confusion matrix."""
    cm = _cm(y, y_pred)
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]
    return [_safe(TP, TP + FN), _safe(FP, FP + TN)]


def balanced_accuracy(y, y_pred):
    """Reported alongside accuracy because the corpus is 29/21: plain accuracy
    flatters a classifier that leans on the majority class."""
    m = evaluation_metrics(y, y_pred)
    return float(np.nanmean([m[1], m[2]]))


def self_test():
    y = np.array([0] * 15 + [1] * 16)
    checks = [
        ("perfect", y.copy(), 1.0),
        ("inverted", 1 - y, 0.0),
        ("all-zeros", np.zeros_like(y), 15 / 31),
    ]
    ok = True
    for name, pred, want in checks:
        got = evaluation_metrics(y, pred)[0]
        same = abs(got - want) < 1e-9
        ok &= same
        print(f"  {name:<10} expected {want:.4f}  got {got:.4f}  "
              f"{'OK' if same else 'FAIL'}")
    # determinism: a metric must not change between identical calls
    reps = {round(evaluation_metrics(y, y.copy())[0], 12) for _ in range(50)}
    print(f"  deterministic over 50 calls: {'OK' if len(reps) == 1 else 'FAIL'}")
    ok &= len(reps) == 1
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
