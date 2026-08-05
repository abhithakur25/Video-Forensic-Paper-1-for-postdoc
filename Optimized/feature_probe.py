"""How much class signal do the extracted features actually carry?

Every deep model in this project collapses onto a single class, which could
mean the architectures are wrong, the training budget is too small, or the
features simply do not separate authentic from forged video. This settles it
cheaply: strong, well-regularised classical classifiers on every available
feature representation, scored by repeated stratified cross-validation.

Classical models are the right instrument here. With 50 samples they are not
starved of data the way a 3D CNN is, they cannot silently collapse without
that showing up in balanced accuracy, and permutation testing gives a null
distribution to compare against.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

P = Path(__file__).resolve().parents[1]
SEED = 1234


def representations():
    import pickle
    with open(P / "Features" / "Features.pkl", "rb") as f:
        d = pickle.load(f)
    lab = np.asarray(d["labels"]).astype(int)

    c1 = np.asarray(d["comparative1"], dtype=np.float32)   # (50,128,128,10)
    c4 = np.asarray(d["comparative4"], dtype=np.float32)   # (50,10,12)
    pr = np.asarray(d["proposed"], dtype=np.float32)       # (50,10,128,128,12)

    reps = {
        # channel-wise summary statistics rather than raw pixels: 16k raw
        # dimensions on 50 samples is hopeless for any classifier.
        "comparative1 (chan mean+std)":
            np.concatenate([c1.mean((1, 2)), c1.std((1, 2))], axis=1),
        "comparative4 (GLCM stats, flat)":
            c4.reshape(len(c4), -1),
        "proposed (chan mean+std)":
            np.concatenate([pr.mean((1, 2, 3)), pr.std((1, 2, 3))], axis=1),
        "proposed (per-frame chan mean)":
            pr.mean((2, 3)).reshape(len(pr), -1),
    }
    cache = P / "Optimized" / "cache"
    for f in sorted(cache.glob("emb_*.npy")):
        reps[f"{f.stem[4:]} embedding"] = np.load(f)
    return reps, lab


def evaluate(X, y, n_rep=5):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, C=0.1,
                                                   class_weight="balanced")),
        "svm-rbf": make_pipeline(StandardScaler(),
                                 SVC(C=1.0, class_weight="balanced")),
        "rf": RandomForestClassifier(n_estimators=400, random_state=SEED,
                                     class_weight="balanced"),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep,
                                 random_state=SEED)
    out = {}
    for name, m in models.items():
        s = cross_val_score(m, X, y, cv=cv, scoring="balanced_accuracy")
        out[name] = (float(s.mean()), float(s.std()))
    return out


def permutation_null(X, y, n=200):
    """Balanced accuracy achievable on shuffled labels - the noise floor."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000, C=0.1,
                                         class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    vals = []
    for _ in range(n):
        yp = rng.permutation(y)
        vals.append(cross_val_score(m, X, yp, cv=cv,
                                    scoring="balanced_accuracy").mean())
    return np.asarray(vals)


def main():
    reps, y = representations()
    print(f"50 samples, class counts {np.bincount(y).tolist()}\n")
    print(f"{'representation':<34} {'dim':>6}  " +
          "  ".join(f"{m:>13}" for m in
                    ["majority", "logreg", "svm-rbf", "rf"]))
    print("-" * 100)
    results = {}
    for name, X in reps.items():
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        r = evaluate(X, y)
        results[name] = r
        print(f"{name:<34} {X.shape[1]:>6}  " +
              "  ".join(f"{r[m][0]*100:6.2f}±{r[m][1]*100:5.2f}"
                        for m in ["majority", "logreg", "svm-rbf", "rf"]))

    best_rep = max(results, key=lambda k: results[k]["logreg"][0])
    print(f"\nPermutation test on '{best_rep}' (logreg, 200 label shuffles):")
    null = permutation_null(np.nan_to_num(
        np.asarray(reps[best_rep], dtype=np.float64)), y)
    obs = results[best_rep]["logreg"][0]
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    print(f"  observed balanced accuracy {obs*100:.2f}%")
    print(f"  null: mean {null.mean()*100:.2f}%  95th pct "
          f"{np.percentile(null, 95)*100:.2f}%  max {null.max()*100:.2f}%")
    print(f"  p = {p:.3f}")
    print(f"  -> {'signal above chance' if p < 0.05 else 'NOT distinguishable from chance'}")

    (P / "Optimized" / "feature_probe.json").write_text(json.dumps({
        "class_counts": np.bincount(y).tolist(),
        "results": {k: {m: v for m, v in r.items()} for k, r in results.items()},
        "permutation": {"representation": best_rep, "observed": obs,
                        "null_mean": float(null.mean()),
                        "null_p95": float(np.percentile(null, 95)),
                        "p_value": p},
    }, indent=2), encoding="utf-8")
    print("\nwrote Optimized/feature_probe.json")


if __name__ == "__main__":
    main()
