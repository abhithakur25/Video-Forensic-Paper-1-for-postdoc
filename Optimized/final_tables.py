"""Full metric tables by training percentage for Paper 1.

Reports accuracy, sensitivity, specificity, precision, recall and F1 for every
training percentage from 40% to 90%, for:

  * the best honest pipeline found by the nested-CV search (L1 logistic
    regression on temporal-delta statistics of the 'proposed' tensor);
  * a few strong alternatives, for context;
  * the paper's own models, taken from Analysis1/TRUE.

Splits use the same deterministic per-class prefix rule as
SubFunctions/Analysis.py, so the rows line up with the paper's tables.

Hyper-parameters are chosen by cross-validation INSIDE the training split
only. The test split is never seen during fitting or selection. This is the
distinction that matters against Paper 2's SubFunctions/Optimization.py, which
receives x_test and y_test in its constructor and searches model weights to
maximise the score on them directly.

Recall and sensitivity are the same quantity (TP / (TP + FN)); both columns are
printed because both were requested.
"""
import json
import os
import sys
import warnings
from pathlib import Path

# joblib/loky workers re-import this module in a fresh interpreter and do not
# inherit filterwarnings(). Without this they flood stderr with SelectKBest
# constant-feature warnings, and on Windows a backed-up stderr pipe surfaces
# as OSError(28, 'No space left on device') from the worker's flush on exit.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "Optimized"))
SEED = 1234
PCTS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def log(m):
    print(m, flush=True)


def temporal_delta(pr):
    """The winning representation: |frame[t+1] - frame[t]| summarised per
    channel. 324 dims for a (10, 128, 128, 12) tensor."""
    d = np.abs(np.diff(pr, axis=1))
    n = len(pr)
    return np.concatenate([d.mean((2, 3)).reshape(n, -1),
                           d.std((2, 3)).reshape(n, -1),
                           d.max(axis=(2, 3)).reshape(n, -1)], 1)


def grid_pool_mean_frame(pr, grids=(1, 2, 4)):
    out = []
    for i in range(len(pr)):
        x = pr[i].mean(0)
        H, W, _ = x.shape
        v = []
        for g in grids:
            hs, ws = H // g, W // g
            for a in range(g):
                for b in range(g):
                    v.append(x[a*hs:(a+1)*hs, b*ws:(b+1)*ws].mean((0, 1)))
        out.append(np.concatenate(v))
    return np.stack(out)


def split_indices(labels, train_size):
    tr, te = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        cut = int(len(idx) * train_size)
        tr.extend(idx[:cut])
        te.extend(idx[cut:])
    return np.array(tr), np.array(te)


def metrics(y, yp):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yp, labels=[0, 1])
    TP, FN, FP, TN = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    f = lambda a, b: (float(a) / float(b)) if b else float("nan")
    sen = f(TP, TP + FN)
    return {
        "Accuracy": f(TP + TN, TP + TN + FP + FN),
        "Sensitivity": sen,
        "Specificity": f(TN, TN + FP),
        "Precision": f(TP, TP + FP),
        "Recall": sen,
        "F1": f(2 * TP, 2 * TP + FP + FN),
        "Balanced": np.nanmean([sen, f(TN, TN + FP)]),
    }


def candidates():
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def pipe(clf):
        return Pipeline([("sc", StandardScaler()), ("sel", "passthrough"),
                         ("clf", clf)])

    red = ["passthrough", SelectKBest(f_classif, k=10),
           SelectKBest(f_classif, k=25), PCA(n_components=8, random_state=SEED),
           PCA(n_components=20, random_state=SEED)]
    return {
        "logreg-l1": (pipe(LogisticRegression(max_iter=5000, penalty="l1",
                                              solver="liblinear",
                                              class_weight="balanced")),
                      {"sel": red, "clf__C": [0.01, 0.1, 1, 10]}),
        "svm-rbf": (pipe(SVC(class_weight="balanced")),
                    {"sel": red, "clf__C": [0.1, 1, 10],
                     "clf__gamma": ["scale", 0.01, 0.001]}),
        "extra-trees": (pipe(ExtraTreesClassifier(n_estimators=300,
                                                  random_state=SEED,
                                                  class_weight="balanced")),
                        {"sel": red, "clf__max_depth": [None, 3, 5]}),
    }


def md_table(rows, head):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i])
                                       for i, c in enumerate(r)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
                     + [line(r) for r in rows])


def main():
    import pickle
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    with open(P / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    y = np.asarray(data["labels"]).astype(int)
    pr = np.asarray(data["proposed"], dtype=np.float32)

    reps = {"temporal delta stats": temporal_delta(pr),
            "mean-frame grid pool": grid_pool_mean_frame(pr)}
    cands = candidates()
    COLS = ["Accuracy", "Sensitivity", "Specificity", "Precision", "Recall",
            "F1", "Balanced"]
    out, store = [], {}

    out.append("# Paper 1 — full metric tables by training percentage\n")
    out.append("All values are percentages, measured with a real confusion "
               "matrix (`Optimized/metrics_fixed.py`). Hyper-parameters are "
               "selected by 4-fold cross-validation **inside the training "
               "split only**; the test split is never used for fitting or "
               "selection.\n")
    out.append("Recall and Sensitivity are the same quantity, TP/(TP+FN); "
               "both columns appear because both were requested.\n")

    for rname, X in reps.items():
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        for cname, (est, grid) in cands.items():
            rows, per = [], []
            for pct in PCTS:
                tr, te = split_indices(y, pct)
                inner = StratifiedKFold(4, shuffle=True, random_state=SEED)
                gs = GridSearchCV(est, grid, scoring="balanced_accuracy",
                                  cv=inner, n_jobs=4, refit=True)
                gs.fit(X[tr], y[tr])
                m = metrics(y[te], gs.predict(X[te]))
                per.append(m)
                rows.append([f"{int(pct*100)}%", f"{len(tr)}/{len(te)}"]
                            + [f"{m[c]*100:.2f}" for c in COLS])
            mean = ["**Mean**", ""] + [
                f"{np.nanmean([p[c] for p in per])*100:.2f}" for c in COLS]
            rows.append(mean)
            store[f"{cname} on {rname}"] = per
            out.append(f"\n## {cname} on {rname}\n")
            out.append(md_table(rows, ["Training %", "Train/Test"] + COLS))
            log(f"{cname:<12} on {rname:<22} "
                f"mean bal-acc {np.nanmean([p['Balanced'] for p in per])*100:.2f}%")

    # ------------------------------------------------- the paper's own models
    tdir = P / "Analysis1" / "TRUE"
    if (tdir / "run_manifest.json").exists():
        man = json.loads((tdir / "run_manifest.json").read_text("utf-8"))
        done = [int(round(p*100)) for p in man["train_pcts"]]
        out.append("\n\n## The paper's own models, correctly scored\n")
        out.append(f"Splits completed: {', '.join(f'{d}%' for d in done)}. "
                   "Columns are means over those splits. Array columns are "
                   "ACC, SEN, SPE, PRE, F1, BAL.\n")
        rows = []
        for f in sorted(tdir.glob("*.npy")):
            a = np.load(f)
            g = lambda i: ("—" if np.all(np.isnan(a[:, i]))
                           else f"{np.nanmean(a[:, i])*100:.2f}")
            rows.append([f.stem, g(0), g(1), g(2), g(3), g(1), g(4), g(5)])
        out.append(md_table(rows, ["Model", "Accuracy", "Sensitivity",
                                   "Specificity", "Precision", "Recall",
                                   "F1", "Balanced"]))

    txt = "\n".join(out)
    (P / "Optimized" / "FINAL_TABLES.md").write_text(txt, encoding="utf-8")
    (P / "Optimized" / "final_tables.json").write_text(
        json.dumps({k: [{c: v[c] for c in COLS} for v in vs]
                    for k, vs in store.items()}, indent=2), encoding="utf-8")
    log("\nwrote Optimized/FINAL_TABLES.md")


if __name__ == "__main__":
    main()
