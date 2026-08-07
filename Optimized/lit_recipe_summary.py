"""Consolidate the literature-recipe experiments and plot the result.

Runs every variant of the frame-level recipe on the same folds, then runs the
same features under a frame-level split, which leaks. Writes
Optimized/frame_level_summary.json and fig19.

The figure exists to make one comparison unmissable: identical features,
identical model, identical code, and the only difference is whether the split
respects video boundaries. That difference is worth 30 percentage points here,
and it is the mechanism by which a 95 % number appears on a corpus that cannot
support one.
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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from frame_level_lit import (RULES, leakage_demo, per_frame_embeddings,  # noqa
                             run_once, score)

P = Path(__file__).resolve().parents[1]
OUT = P / "Results" / "Genuine"
ACCENT = "#1F4E79"


def main():
    fz = np.load(P / "Optimized" / "cache" / "folds.npz")
    y = fz["y"].astype(int)
    folds = [(fz[f"tr{i}"], fz[f"te{i}"]) for i in range(int(fz["n_folds"]))]

    rgb = per_frame_embeddings("xception", True, "rgb")
    dif = per_frame_embeddings("xception", True, "diff")
    m = min(rgb.shape[1], dif.shape[1])
    both = np.concatenate([rgb[:, :m], dif[:, :m]], axis=2)

    variants = [("Xception RGB frames", rgb),
                ("Xception frame differences", dif),
                ("Xception RGB + differences", both)]

    correct, leaky = {}, {}
    for label, emb in variants:
        t0 = time.time()
        oof, _ = run_once(emb, y, folds)
        best = max(RULES, key=lambda r: score(oof[r], y)["bal"])
        correct[label] = {**score(oof[best], y), "rule": best,
                          "seconds": round(time.time() - t0, 1)}
        leaky[label] = leakage_demo(emb, y)
        print(f"{label:<30} video-grouped {correct[label]['bal']:6.2f}%   "
              f"frame-split(LEAKY) {leaky[label]['bal']:6.2f}%")

    # The reference points this study already established.
    v2 = json.loads((P / "Optimized" / "optimize_v2.json")
                    .read_text("utf-8"))["winner"]
    roc = json.loads((P / "Optimized" / "roc_confusion.json")
                     .read_text("utf-8"))
    ref = roc["curves"]["temporal delta stats (best honest pipeline)"]

    summary = {
        "question": "what do the published FaceForensics++ recipes give on "
                    "this corpus?",
        "recipe_from_literature": {
            "frame_level_training": True,
            "video_level_aggregation": ["mean probability", "mean log-odds",
                                        "majority vote"],
            "backbone": "Xception, ImageNet, 299x299 face crop",
            "test_time_augmentation": "horizontal flip",
            "split": "by video - frames of one video never straddle a fold",
        },
        "video_grouped_correct": correct,
        "frame_split_leaky": leaky,
        "reference_this_study": {
            "temporal delta stats + L1 logreg (nested fold mean)":
                v2["nested_bal_acc"] * 100,
            "temporal delta stats (pooled out-of-fold)":
                ref["balanced_accuracy"] * 100,
            "pooled out-of-fold AUC": ref["auc"],
        },
        "conclusion": "the literature's frame-level recipe, applied faithfully "
                      "with a video-grouped split, scores below the temporal "
                      "descriptor already measured here. The same features "
                      "under a frame-level split exceed 90%, and that gap is "
                      "entirely leakage.",
    }
    f = P / "Optimized" / "frame_level_summary.json"
    f.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {f.name}")

    # ---------------------------------------------------------------- figure
    labels = [v[0] for v in variants]
    a = [correct[k]["bal"] for k in labels]
    b = [leaky[k]["bal"] for k in labels]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, a, w, label="split by video (correct)",
                color="#2E75B6", edgecolor="#37474F", linewidth=0.5)
    b2 = ax.bar(x + w / 2, b, w, label="split by frame (leaks)",
                color="#C62828", edgecolor="#37474F", linewidth=0.5)
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ref_bal = ref["balanced_accuracy"] * 100
    ax.axhline(ref_bal, color="#2E7D32", ls="-", lw=1.4)
    ax.text(len(labels) - 0.45, ref_bal + 1.2,
            f"temporal delta descriptor already measured here "
            f"({ref_bal:.2f}%)", fontsize=8, color="#2E7D32", ha="right")
    ax.axhline(50, color="#37474F", ls="--", lw=1.0)
    ax.text(-0.45, 51.2, "constant classifier", fontsize=8, color="#37474F")
    ax.axhline(95, color="#8E24AA", ls=":", lw=1.3)
    ax.text(-0.45, 96.2, "95% target", fontsize=8, color="#8E24AA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("The published frame-level recipe on this corpus\n"
                 "Identical features and model in each pair — only the split "
                 "differs", fontsize=11.5, color=ACCENT, weight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", ls=":", lw=0.6, color="#B0BEC5")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig19_literature_recipe_and_leakage.png",
                bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print("wrote fig19_literature_recipe_and_leakage.png")


if __name__ == "__main__":
    main()
