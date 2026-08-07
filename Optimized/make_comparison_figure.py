"""Figure 13: every method measured in this study on one axis.

The comparison bar chart for Section 6. Bars are mean balanced accuracy, so the
constant-classifier line sits at exactly 50 and a bar at that height means the
model answered one class for every input rather than that it was half right.
The permutation null's 95th percentile is drawn as the line a bar must clear to
carry any evidence of detection.

Colour encodes only whether a bar clears that line, so the figure cannot be
read as more favourable than the statistics support.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from article_data import ORDER, load, mean_bal, pretty   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
OUT = P / "Results" / "Genuine"

C_GOOD = "#2E7D32"
C_MID = "#8FAF6E"
C_DEGEN = "#B0BEC5"
C_BAD = "#C62828"


def main():
    d = load()
    null95 = d["v2"]["winner"]["null_p95"] * 100

    # (label, value, protocol, permutation p-value or None)
    # Only the nested-CV pipeline was permutation-tested, so only it can be
    # coloured as carrying evidence. Everything else is coloured purely by
    # its position relative to the constant classifier, which is a
    # description rather than a claim.
    rows = [("Temporal deltas + L1 logistic",
             d["v2"]["winner"]["nested_bal_acc"] * 100, "nested CV",
             d["v2"]["winner"]["p_value"])]
    rows += [(pretty(m), mean_bal(d["kf"][m]), "k-fold", None)
             for m in ORDER]
    rows.append(("STIL TIM + ISM (imported)",
                 d["stil"]["pooled_balanced_accuracy"] * 100, "nested CV",
                 None))
    rows.append(("BiLSTM + GBM (companion)",
                 d["paper2"]["BiLSTMGBM"][5] * 100, "5-fold", None))
    rows.append(("Constant 'authentic' baseline", 50.0, "—", None))
    rows.sort(key=lambda t: t[1])

    labels = [f"{n}  [{p}]" for n, _, p, _ in rows]
    vals = np.array([v for _, v, _, _ in rows])
    colors = []
    for (_, v, _, pv) in rows:
        if pv is not None and pv < 0.05 and v > null95:
            colors.append(C_GOOD)
        elif v > 50.0:
            colors.append(C_MID)
        elif v == 50.0:
            colors.append(C_DEGEN)
        else:
            colors.append(C_BAD)

    fig, ax = plt.subplots(figsize=(10.2, 6.4), dpi=200)
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, edgecolor="#37474F", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Mean balanced accuracy (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.axvline(50.0, color="#37474F", lw=1.2, ls="--")
    ax.axvline(null95, color="#1B5E20", lw=1.2, ls=":")
    top = len(rows) - 0.1
    ax.text(49.4, top + 0.45, "constant classifier (50%)", fontsize=7.5,
            color="#37474F", ha="right", va="center")
    ax.text(null95 + 0.8, top + 0.45,
            f"permutation null, 95th pct ({null95:.1f}%), nested-CV entries",
            fontsize=7.5, color="#1B5E20", ha="left", va="center")
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_ylim(-1.0, top + 1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#37474F", lw=0.6)
               for c in (C_GOOD, C_MID, C_DEGEN, C_BAD)]
    ax.legend(handles,
              ["clears its own permutation null (p < 0.05)",
               "above the constant classifier, not permutation-tested",
               "exactly 50.00 — answered one class for every input",
               "below the constant classifier"],
              fontsize=7.5, loc="lower right", framealpha=0.95)
    ax.set_title("Every method measured in this study, on identical splits",
                 fontsize=11.5, color="#1F4E79", weight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", ls=":", lw=0.6, color="#B0BEC5")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig13_comparison_bar.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"fig13_comparison_bar.png  ({len(rows)} methods, "
          f"{sum(1 for c in colors if c == C_GOOD)} above the null)")


if __name__ == "__main__":
    main()
