"""Bar charts for every metric in every results table, plus ROC.

Generates, in the order the tables appear in the paper:

  fig14  accuracy / precision / recall / F1, grouped, training-percentage means
  fig15  the same four metrics, k-fold means
  fig16  each metric against training percentage, one panel per metric
  fig17  each metric against k, one panel per metric
  fig18  ROC: the two measured curves with their AUC, and every model in the
         cohort plotted as its single operating point

A note on fig18, because it is the one place where the honest figure differs
from the one usually printed. A ROC curve needs a continuous score. The sweep
and k-fold runs store arg-max predictions, not probabilities, so each model
yields exactly one (FPR, TPR) point and no curve. Drawing a smooth curve
through one point - which is what a great deal of published work does - invents
the shape between the ends. This figure plots the two curves that really exist,
from the nested-CV pipelines that stored probabilities, and puts every other
model on the same axes as the point it actually is.

A note on 'recall'. The released evaluation code treats class 0 (authentic) as
positive, so its sensitivity is the recall of the authentic class and its
specificity is the recall of the forged class. Both are plotted. Reporting only
the first is what lets a model that never reports a forgery look like a
detector.

Progress is printed as a bar so a long run is watchable.

    python Optimized/make_metric_charts.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from article_data import (ACC, BAL, F1, ORDER, PRE, SEN, SPE, load,  # noqa: E402
                          pretty)

# Jupyter replaces sys.stdout with an ipykernel OutStream, which has no
# reconfigure(); this module is imported by Paper1_Results.ipynb.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
OUT = P / "Results" / "Genuine"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#1F4E79"
BAR = ["#2E75B6", "#7FA9D4", "#F2A65A", "#8FBF6E"]
C_DEGEN = "#C00000"
plt.rcParams["font.family"] = "DejaVu Sans"

# (column index, label) in table order
METRICS = [(ACC, "Accuracy"), (PRE, "Precision"),
           (SEN, "Recall (authentic)"), (SPE, "Recall (forged)"),
           (F1, "F1"), (BAL, "Balanced accuracy")]
GROUPED = [(ACC, "Accuracy"), (PRE, "Precision"),
           (SEN, "Recall (authentic)"), (F1, "F1")]


class Progress:
    """A carriage-return progress bar, so a slow run is watchable."""

    def __init__(self, total, width=42):
        self.total, self.width, self.n = total, width, 0
        self.t0 = time.time()

    def step(self, label=""):
        self.n += 1
        frac = self.n / self.total
        fill = int(self.width * frac)
        el = time.time() - self.t0
        eta = el / frac - el if frac else 0
        sys.stdout.write(
            f"\r  [{'█' * fill}{'·' * (self.width - fill)}] "
            f"{self.n:2d}/{self.total}  {frac * 100:5.1f}%  "
            f"eta {eta:4.1f}s  {label:<34}")
        sys.stdout.flush()
        if self.n == self.total:
            sys.stdout.write(f"\r  [{'█' * self.width}] {self.total}/"
                             f"{self.total}  100.0%  done in {el:.1f}s"
                             f"{' ' * 34}\n")


def _save(fig, name, pr):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    pr.step(name)


def _order(d, key):
    return sorted(ORDER, key=lambda m: -np.nanmean(d[key][m][:, BAL]))


def grouped_bars(d, key, title, name, pr):
    """Four metrics side by side per model, ordered by balanced accuracy."""
    names = _order(d, key)
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    x = np.arange(len(names))
    w = 0.8 / len(GROUPED)
    for i, (col, lab) in enumerate(GROUPED):
        vals = [np.nanmean(d[key][m][:, col]) for m in names]
        vals = [0.0 if np.isnan(v) else v for v in vals]
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, width=w, label=lab,
                   color=BAR[i], edgecolor="#37474F", linewidth=0.4)
        ax.bar_label(b, fmt="%.0f", fontsize=5.6, padding=1)
    # Recall-on-forged is zero in two quite different ways and the difference
    # matters: a model that never predicts 'forged' at all is flattered by
    # accuracy, precision and F1, whereas one that predicts it and is always
    # wrong is inverting the decision. Neither is visible in the four bars.
    for xi, m in zip(x, names):
        spe = np.nanmean(d[key][m][:, SPE])
        sen = np.nanmean(d[key][m][:, SEN])
        if spe == 0 and sen == 100:
            ax.text(xi, -8.0, "never predicts\n'forged'", ha="center",
                    fontsize=5.8, color=C_DEGEN, style="italic")
        elif spe == 0:
            ax.text(xi, -8.0, "catches no\nforgeries", ha="center",
                    fontsize=5.8, color=C_DEGEN, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(m) for m in names], rotation=32, ha="right",
                       fontsize=8)
    ax.set_ylabel("Percent")
    ax.set_ylim(-15, 122)
    ax.axhline(0, color="#37474F", lw=0.8)
    ax.set_title(title, fontsize=11.5, color=ACCENT, weight="bold")
    ax.legend(fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), framealpha=0.95)
    ax.grid(axis="y", ls=":", lw=0.6, color="#B0BEC5")
    ax.set_axisbelow(True)
    _save(fig, name, pr)


def metric_panels(d, key, axis_vals, axis_label, title, name, pr):
    """One panel per metric, every model as a line across the axis."""
    names = _order(d, key)
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True)
    for ax, (col, lab) in zip(axes.ravel(), METRICS):
        for j, m in enumerate(names):
            a = d[key][m]
            n = min(len(axis_vals), a.shape[0])
            ax.plot(axis_vals[:n], a[:n, col], marker="o", ms=3.2, lw=1.1,
                    color=cmap(j % 20), label=pretty(m))
        ax.set_title(lab, fontsize=9.5, color=ACCENT)
        ax.set_ylim(-4, 104)
        ax.grid(ls=":", lw=0.6, color="#CFD8DC")
        ax.set_axisbelow(True)
        ax.set_xticks(axis_vals)
    for ax in axes[1]:
        ax.set_xlabel(axis_label, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Percent", fontsize=9)
    h, la = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, la, fontsize=7.2, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(title, fontsize=12, color=ACCENT, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, name, pr)


def roc_figure(d, pr):
    """The two measured curves, plus every model as the one point it is."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.6))

    # --- left: the curves that actually exist
    styles = [("temporal delta stats (best honest pipeline)",
               "Temporal delta statistics", "#2E7D32", "-"),
              ("per-frame mean+std (time-collapsed reference)",
               "Time-collapsed reference", "#C62828", "--")]
    for k, lab, col, ls in styles:
        c = d["roc"]["curves"][k]
        axL.plot(c["fpr"], c["tpr"], color=col, ls=ls, lw=2.0,
                 label=f"{lab}  (AUC = {c['auc']:.4f})")
        axL.fill_between(c["fpr"], c["tpr"], alpha=0.07, color=col)
    perm = d["roc"]["auc_permutation"]
    axL.plot([0, 1], [0, 1], color="#78909C", ls=":", lw=1.2,
             label="chance (AUC = 0.5000)")
    axL.set_xlabel("False positive rate")
    axL.set_ylabel("True positive rate")
    axL.set_title("ROC and AUC, out-of-fold over all 50 videos\n"
                  f"permutation null: mean {perm['null_mean']:.4f}, "
                  f"95th pct {perm['null_p95']:.4f}, "
                  f"p = {perm['p_value']:.4f}",
                  fontsize=10.5, color=ACCENT, weight="bold")
    axL.legend(fontsize=8, loc="lower right")
    axL.grid(ls=":", lw=0.6, color="#CFD8DC")
    axL.set_xlim(-0.02, 1.02)
    axL.set_ylim(-0.02, 1.02)

    # --- right: hard predictions give one point each, not a curve.
    # The degenerate models land on exactly the same coordinates, so plotting
    # them naively hides all but the last one drawn - which would conceal the
    # single most important fact in the figure. Coincident models are grouped
    # and the group is labelled with its members.
    cmap = plt.get_cmap("tab20")
    pts = {}
    for j, m in enumerate(_order(d, "kf")):
        a = d["kf"][m]
        tpr = np.nanmean(a[:, SEN]) / 100.0        # class 0 as positive
        fpr = 1.0 - np.nanmean(a[:, SPE]) / 100.0
        # 2 dp is the resolution of the plotted marker: anything closer than
        # that overlaps on the page whether or not the numbers are equal.
        pts.setdefault((round(fpr, 2), round(tpr, 2)), []).append((j, m))
    for (fpr, tpr), members in pts.items():
        j, m = members[0]
        lab = pretty(m) if len(members) == 1 else \
            f"{pretty(m)}  (+{len(members) - 1} overlapping)"
        axR.scatter(fpr, tpr, s=58 + 26 * (len(members) - 1),
                    color=cmap(j % 20), edgecolor="#37474F", linewidth=0.6,
                    zorder=3, label=lab)
        if len(members) > 1:
            axR.annotate("\n".join(pretty(x[1]) for x in members),
                         (fpr, tpr), textcoords="offset points",
                         xytext=(-12, -14), ha="right", va="top",
                         fontsize=6.4, color="#C62828",
                         arrowprops=dict(arrowstyle="-", lw=0.6,
                                         color="#C62828"))
    axR.plot([0, 1], [0, 1], color="#78909C", ls=":", lw=1.2)
    axR.text(0.52, 0.46, "chance", fontsize=8, color="#78909C", rotation=39)
    axR.scatter([0], [1], marker="*", s=170, color="#2E7D32", zorder=4)
    axR.text(0.02, 0.955, "perfect", fontsize=8, color="#2E7D32")
    axR.set_xlabel("False positive rate  (1 − recall on forged)")
    axR.set_ylabel("True positive rate  (recall on authentic)")
    axR.set_title("Cohort operating points, k-fold means\n"
                  "one point per model: the runs stored arg-max predictions, "
                  "not scores", fontsize=10.5, color=ACCENT, weight="bold")
    axR.legend(fontsize=6.6, ncol=2, loc="lower right", framealpha=0.95)
    axR.grid(ls=":", lw=0.6, color="#CFD8DC")
    axR.set_xlim(-0.02, 1.02)
    axR.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, "fig18_roc_and_operating_points.png", pr)


def main():
    print(f"reading measured results from Analysis1/ and Optimized/")
    d = load()
    print(f"  {len(ORDER)} models · train% {d['pcts']} · k {d['ks']}\n")
    print("generating charts")
    pr = Progress(5)
    grouped_bars(d, "sweep",
                 "Accuracy, precision, recall and F1 — means over the "
                 "training-percentage sweep",
                 "fig14_metrics_sweep_grouped.png", pr)
    grouped_bars(d, "kf",
                 f"Accuracy, precision, recall and F1 — means over k = "
                 f"{', '.join(str(k) for k in d['ks'])}",
                 "fig15_metrics_kfold_grouped.png", pr)
    metric_panels(d, "sweep", d["pcts"], "Training data (%)",
                  "Every metric against training percentage",
                  "fig16_metric_panels_sweep.png", pr)
    metric_panels(d, "kf", d["ks"], "k",
                  "Every metric against k", "fig17_metric_panels_kfold.png",
                  pr)
    roc_figure(d, pr)

    man = json.loads((P / "Analysis1" / "TRUE_KF" / "run_manifest.json")
                     .read_text("utf-8"))
    print(f"\nwrote 5 charts to {OUT.relative_to(P)}")
    print(f"k-fold state at generation time: k = {man['k_values']}")


if __name__ == "__main__":
    main()
