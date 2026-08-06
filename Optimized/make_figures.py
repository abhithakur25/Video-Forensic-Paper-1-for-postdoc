"""Generate every results figure from the measured arrays.

The figures shipped with the original delivery were plotted from the tampered
metric and were removed (see PROVENANCE.md). These replace them, drawn only
from files produced by runs scored with metrics_fixed.py.

Every plot carries the reference lines that make it readable on this corpus:
chance at 50% balanced accuracy, and the majority-class baseline at 58%
accuracy. A bar that reaches 58% accuracy while sitting on the 50% balanced
line is a constant classifier, and the figures are drawn so that is visible
rather than hidden.

Output: Results/Genuine/*.png at 150 dpi.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
OUT = P / "Results" / "Genuine"
DPI = 150

CHANCE = 50.0
MAJORITY_ACC = 58.0
C_GOOD, C_DEGEN, C_REF = "#2E75B6", "#C00000", "#7F7F7F"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    "axes.axisbelow": True,
})


def log(m):
    print(m, flush=True)


def load_arrays(sub):
    d = P / "Analysis1" / sub
    if not (d / "run_manifest.json").exists():
        return {}, {}
    man = json.loads((d / "run_manifest.json").read_text("utf-8"))
    return {f.stem: np.load(f) for f in sorted(d.glob("*.npy"))}, man


def load_json(n):
    f = P / "Optimized" / n
    return json.loads(f.read_text("utf-8")) if f.exists() else None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    log(f"  {name}  ({p.stat().st_size/1024:.0f} KB)")
    return name


def fig_model_bars(tr, made):
    """Balanced accuracy and accuracy per model, sweep means."""
    names = sorted(tr, key=lambda k: -np.nanmean(tr[k][:, 5]))
    bal = [np.nanmean(tr[n][:, 5]) * 100 for n in names]
    acc = [np.nanmean(tr[n][:, 0]) * 100 for n in names]
    degen = [abs(b - CHANCE) < 1e-9 for b in bal]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(names))
    ax.bar(x, bal, color=[C_DEGEN if d else C_GOOD for d in degen],
           edgecolor="black", linewidth=0.4)
    ax.axhline(CHANCE, color=C_REF, ls="--", lw=1.2,
               label="chance / constant classifier (50.00%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylabel("Mean balanced accuracy (%)")
    ax.set_title("Balanced accuracy by model, mean over six training "
                 "percentages\n(red = degenerate: one label for every input)")
    ax.set_ylim(0, 75)
    for xi, v in zip(x, bal):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)
    ax.legend(loc="upper right", fontsize=8)
    made.append(save(fig, "fig01_balanced_accuracy_by_model.png"))

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x, acc, color=C_GOOD, edgecolor="black", linewidth=0.4)
    ax.axhline(MAJORITY_ACC, color=C_DEGEN, ls="--", lw=1.2,
               label="majority-class baseline (58.00%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_title("Accuracy by model, mean over six training percentages\n"
                 "Bars at or below the dashed line carry no information")
    ax.set_ylim(0, 75)
    for xi, v in zip(x, acc):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)
    ax.legend(loc="upper right", fontsize=8)
    made.append(save(fig, "fig02_accuracy_by_model.png"))


def fig_training_pct(tr, man, made):
    pcts = [int(p * 100) for p in man["train_pcts"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for n in sorted(tr, key=lambda k: -np.nanmean(tr[k][:, 0])):
        ax.plot(pcts, tr[n][:, 0] * 100, marker="o", ms=3.5, lw=1.1, label=n)
    ax.axhline(MAJORITY_ACC, color=C_DEGEN, ls="--", lw=1.2,
               label="majority baseline")
    ax.set_xlabel("Training percentage (%)   —   test set 31 videos → 6")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs training percentage\n"
                 "One misclassification is worth 3.23 pp at 40% and "
                 "16.67 pp at 90%")
    ax.set_xticks(pcts)
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    made.append(save(fig, "fig03_accuracy_vs_training_percentage.png"))


def fig_kfold(kf, kman, made):
    ks = kman["k_values"]
    names = sorted(kf, key=lambda k: -np.nanmean(kf[k][:, 5]))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(names))
    w = 0.8 / max(1, len(ks))
    cmap = plt.get_cmap("Blues")
    for i, k in enumerate(ks):
        ax.bar(x + i * w - 0.4 + w / 2, [kf[n][i, 5] * 100 for n in names],
               width=w, label=f"k={k}", edgecolor="black", linewidth=0.3,
               color=cmap(0.35 + 0.5 * i / max(1, len(ks) - 1)))
    ax.axhline(CHANCE, color=C_DEGEN, ls="--", lw=1.2, label="chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title(f"K-fold balanced accuracy, k = "
                 f"{', '.join(str(k) for k in ks)}\n"
                 f"Test folds hold 5–9 of 50 videos; one error moves a bar "
                 f"11–20 pp")
    ax.legend(fontsize=7, ncol=len(ks) + 1)
    made.append(save(fig, "fig04_kfold_balanced_accuracy.png"))


def fig_roc(roc, made):
    key = "temporal delta stats (best honest pipeline)"
    ref = "per-frame mean+std (time-collapsed reference)"
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for k, col, ls in [(key, C_GOOD, "-"), (ref, C_DEGEN, "--")]:
        c = roc["curves"][k]
        ax.plot(c["fpr"], c["tpr"], color=col, ls=ls, lw=1.8,
                label=f"{k.split(' (')[0]}  (AUC {c['auc']:.4f})")
    ax.plot([0, 1], [0, 1], color=C_REF, ls=":", lw=1.2, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC, out-of-fold over all 50 videos\n"
                 "Collapsing the same tensor over time destroys the signal")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    made.append(save(fig, "fig05_roc_curve.png"))


def fig_confusion(roc, made):
    c = roc["curves"]["temporal delta stats (best honest pipeline)"]
    cm = c["confusion_matrix"]
    M = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(M[i, j]), ha="center", va="center",
                    fontsize=17,
                    color="white" if M[i, j] > M.max() * 0.55 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted\nauthentic", "Predicted\nforged"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True\nauthentic", "True\nforged"])
    ax.set_title(f"Out-of-fold confusion matrix\n"
                 f"acc {c['accuracy']*100:.2f}%  bal {c['balanced_accuracy']*100:.2f}%"
                 f"  AUC {c['auc']:.4f}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    made.append(save(fig, "fig06_confusion_matrix.png"))


def fig_representations(v2, made):
    top = v2["ranking"][:12][::-1]
    lab = [f"{r['representation'][:34]} · {r['model']}" for r in top]
    val = [r["mean"] * 100 for r in top]
    err = [r["std"] * 100 for r in top]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(top))
    ax.barh(y, val, xerr=err, color=C_GOOD, edgecolor="black", linewidth=0.4,
            error_kw=dict(lw=0.8, capsize=2.5, ecolor="#555555"))
    ax.axvline(CHANCE, color=C_REF, ls="--", lw=1.2, label="chance")
    w = v2["winner"]
    ax.axvline(w["null_p95"] * 100, color=C_DEGEN, ls=":", lw=1.4,
               label=f"permutation null 95th pct ({w['null_p95']*100:.2f}%)")
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=6.5)
    ax.set_xlabel("Nested-CV balanced accuracy (%)   ± SD across outer folds")
    ax.set_title(f"Top 12 of {len(v2['ranking'])} representation × model "
                 f"combinations\nOnly the top bar clears the permutation null")
    ax.legend(fontsize=7, loc="lower right")
    made.append(save(fig, "fig07_representation_search.png"))


def fig_method_summary(tr, roc, stil, v2, made):
    """Colour encodes the only criterion that matters here: does the bar clear
    the permutation null? Anything below it is indistinguishable from a model
    trained on shuffled labels, however far above 50% it happens to sit."""
    rows = []
    if roc:
        c = roc["curves"]["temporal delta stats (best honest pipeline)"]
        rows.append(("L1 logreg on\ntemporal deltas",
                     c["balanced_accuracy"] * 100))
    if stil:
        rows.append(("STIL TIM+ISM\n(Tencent TFace)",
                     stil["pooled_balanced_accuracy"] * 100))
    if tr:
        for n, lab in [("SMA-CLMPNet", "SMA-CLMPNet\n(proposed)"),
                       ("MobileNetV3Large", "MobileNetV3-Large\n(frozen)")]:
            if n in tr:
                rows.append((lab, np.nanmean(tr[n][:, 5]) * 100))
    if roc:
        c = roc["curves"]["per-frame mean+std (time-collapsed reference)"]
        rows.append(("Time-collapsed\nreference", c["balanced_accuracy"] * 100))

    null95 = v2["winner"]["null_p95"] * 100 if v2 else CHANCE
    rows = [(lab, v, C_GOOD if v > null95 else C_DEGEN) for lab, v in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], color=[r[2] for r in rows],
           edgecolor="black", linewidth=0.5, width=0.6)
    ax.axhline(CHANCE, color=C_REF, ls="--", lw=1.3, label="chance (50%)")
    if v2:
        ax.axhline(v2["winner"]["null_p95"] * 100, color="#ED7D31", ls=":",
                   lw=1.4, label="permutation null 95th pct")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("Every method tried, on identical data and folds\n"
                 "Blue clears the permutation null; red does not")
    ax.set_ylim(0, 85)
    for xi, r in zip(x, rows):
        ax.text(xi, r[1] + 1.2, f"{r[1]:.2f}", ha="center", fontsize=8.5,
                fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right")
    made.append(save(fig, "fig08_method_comparison.png"))


def fig_ceiling(audit, made):
    if "max_accuracy_any_threshold" not in audit:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    vals = [audit["oof_auc"] * 100,
            audit["max_accuracy_any_threshold"] * 100,
            audit["null_auc_p95"] * 100, 95.0]
    lab = ["Measured\nAUC ×100", "Max accuracy at\nANY threshold",
           "Permutation null\n95th pct (AUC×100)", "Requested\ntarget"]
    col = [C_GOOD, C_GOOD, C_REF, C_DEGEN]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=col, edgecolor="black", linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.5, f"{v:.2f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_title("The attainable-accuracy ceiling on this corpus\n"
                 "Accuracy is bounded by the ROC curve: AUC 0.7307 caps it "
                 "at 74.00%")
    made.append(save(fig, "fig09_accuracy_ceiling.png"))


def main():
    made = []
    tr, man = load_arrays("TRUE")
    kf, kman = load_arrays("TRUE_KF")
    roc, audit = load_json("roc_confusion.json"), load_json("corpus_audit.json")
    stil, v2 = load_json("oof_stil_tim.json"), load_json("optimize_v2.json")

    log("writing figures to Results/Genuine/")
    if tr:
        fig_model_bars(tr, made)
        fig_training_pct(tr, man, made)
    if kf:
        fig_kfold(kf, kman, made)
    if roc:
        fig_roc(roc, made)
        fig_confusion(roc, made)
    if v2:
        fig_representations(v2, made)
    if tr or roc:
        fig_method_summary(tr, roc, stil, v2, made)
    if audit:
        fig_ceiling(audit, made)

    (OUT / "MANIFEST.json").write_text(json.dumps({
        "figures": made,
        "sources": {
            "fig01-03": "Analysis1/TRUE", "fig04": "Analysis1/TRUE_KF",
            "fig05-06": "Optimized/roc_confusion.json",
            "fig07": "Optimized/optimize_v2.json",
            "fig08": "all of the above + Optimized/oof_stil_tim.json",
            "fig09": "Optimized/corpus_audit.json"},
        "scored_with": "Optimized/metrics_fixed.py (real confusion matrix)",
    }, indent=2), encoding="utf-8")
    log(f"{len(made)} figures + MANIFEST.json")


if __name__ == "__main__":
    main()
