"""Figure 20: the frequency representation against the previous best.

The literature search produced one representation that genuinely improves on
what this project had: F3-Net's frequency-aware decomposition, keeping the high
radial band, embedded with the same Xception backbone and concatenated with the
temporal-difference descriptor.

The figure shows why that improvement does not reach the 95 % target. The
ranking improves - the curve moves up and left, and the area under it rises
from 0.7307 to 0.7980 - while the best accuracy reachable at any threshold on
either curve is identical at 74.00 %. On fifty samples the curve is a staircase
of at most fifty steps, so a better ordering of the videos does not necessarily
buy a better cut point.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from freq_ensemble_lit import bal_from_prob, nested_oof   # noqa: E402

P = Path(__file__).resolve().parents[1]
OUT = P / "Results" / "Genuine"
ACCENT = "#1F4E79"


def main():
    from sklearn.metrics import roc_auc_score, roc_curve

    fz = np.load(P / "Optimized" / "cache" / "folds.npz")
    y = fz["y"].astype(int)
    folds = [(fz[f"tr{i}"], fz[f"te{i}"]) for i in range(int(fz["n_folds"]))]
    z = np.load(P / "Optimized" / "cache" / "freq_reps.npz")
    fad = z["FAD high band (Xception)"]
    td = z["Temporal delta stats (Section 5.5)"]

    series = [
        ("Temporal deltas (previous best)", td, "#C62828", "--"),
        ("FAD high band, Xception (new)", fad, "#1565C0", "-."),
        ("FAD high band + temporal deltas", np.concatenate([fad, td], 1),
         "#2E7D32", "-"),
    ]
    p1 = (y == 1).mean()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.4))
    rows = []
    for label, X, col, ls in series:
        p, _ = nested_oof(X, y, folds)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        acc = tpr * p1 + (1 - fpr) * (1 - p1)
        axL.plot(fpr, tpr, color=col, ls=ls, lw=2.0,
                 label=f"{label}  (AUC = {auc:.4f})")
        axR.plot(fpr, acc * 100, color=col, ls=ls, lw=2.0, label=label)
        rows.append((label, auc, acc.max() * 100, bal_from_prob(p, y)))

    axL.plot([0, 1], [0, 1], color="#78909C", ls=":", lw=1.2, label="chance")
    axL.set_xlabel("False positive rate")
    axL.set_ylabel("True positive rate")
    axL.set_title("The ranking improves", fontsize=11, color=ACCENT,
                  weight="bold")
    axL.legend(fontsize=8, loc="lower right")
    axL.grid(ls=":", lw=0.6, color="#CFD8DC")

    axR.axhline(95, color="#8E24AA", ls=":", lw=1.4)
    axR.text(0.02, 96, "95% target", fontsize=8.5, color="#8E24AA")
    axR.axhline(58, color="#37474F", ls="--", lw=1.0)
    axR.text(0.02, 54.5, "majority class (58%)", fontsize=8.5,
             color="#37474F")
    axR.set_xlabel("False positive rate (threshold sweep)")
    axR.set_ylabel("Accuracy at that threshold (%)")
    axR.set_ylim(0, 102)
    axR.set_title("The reachable accuracy does not", fontsize=11,
                  color=ACCENT, weight="bold")
    axR.legend(fontsize=8, loc="lower right")
    axR.grid(ls=":", lw=0.6, color="#CFD8DC")

    fig.suptitle("Frequency-aware decomposition (F3-Net) on this corpus: "
                 "a better ordering, the same ceiling",
                 fontsize=12, color=ACCENT, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig20_frequency_representation.png",
                bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)

    print(f"{'representation':<36}{'AUC':>9}{'max acc':>10}{'BAL':>8}")
    print("-" * 63)
    for label, auc, mx, bal in rows:
        print(f"{label:<36}{auc:9.4f}{mx:10.2f}{bal:8.2f}")
    (P / "Optimized" / "fad_figure.json").write_text(
        json.dumps([{"representation": r[0], "auc": r[1],
                     "max_accuracy_any_threshold": r[2], "bal": r[3]}
                    for r in rows], indent=2), encoding="utf-8")
    print("\nwrote fig20_frequency_representation.png")


if __name__ == "__main__":
    main()
