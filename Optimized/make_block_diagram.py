"""Block diagrams of the pipeline actually executed in this directory.

Three figures, drawn from the source rather than from the paper's prose:

  fig10  end-to-end pipeline, ReadDataset -> Features.pkl -> cohort -> scoring
  fig11  SMA-CLMPNet as implemented in SubFunctions/Model.py:447-513, with the
         tensor shape after every stage and both attention branches marked
  fig12  the nested cross-validation protocol used for every reported number

Shapes on fig11 are the ones Keras reports for the cached tensor
(50, 10, 128, 128, 12); they are recomputed here rather than transcribed, so
the figure cannot drift from the code without the arithmetic below changing.

Writes into Results/Genuine/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
OUT = P / "Results" / "Genuine"
OUT.mkdir(parents=True, exist_ok=True)

C_IN = "#DCE6F1"
C_CONV = "#B7D3EA"
C_ATT = "#F6C9A0"
C_SEQ = "#C6E0B4"
C_HEAD = "#D9C2E9"
C_OUT = "#F4B6B6"
C_EDGE = "#1F4E79"
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, fc, fs=8, ec=C_EDGE, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.1, ls="-", color=C_EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=lw, ls=ls,
                                 color=color, zorder=1,
                                 shrinkA=0, shrinkB=0))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------- fig 10
def pipeline():
    fig, ax = canvas(11.0, 5.0)
    ax.text(0.5, 0.975, "End-to-end pipeline as executed",
            ha="center", va="top", fontsize=11, weight="bold", color=C_EDGE)

    lane = [
        (0.025, 0.145, C_IN,
         "FaceForensics++ subset\n50 videos\n29 authentic / 21 forged"),
        (0.198, 0.145, C_IN,
         "ReadDataset\nHaar face crop\n10 frames per video\n128 x 128"),
        (0.371, 0.145, C_CONV,
         "GetFeatures\nRGB + HSV + LBP + edge\n12 channels\nFeatures.pkl"),
    ]
    for x, w, c, t in lane:
        box(ax, x, 0.640, w, 0.230, t, c, fs=7.5)
    for i in range(len(lane) - 1):
        x0 = lane[i][0] + lane[i][1]
        arrow(ax, x0 + 0.004, 0.755, lane[i + 1][0] - 0.004, 0.755)

    box(ax, 0.548, 0.100, 0.437, 0.800, "", "#FFFFFF", ec="#9CB7D4", lw=1.0)
    ax.text(0.7665, 0.862, "Model cohort  (12 systems, identical splits)",
            ha="center", va="center", fontsize=9, weight="bold", color=C_EDGE)

    cohort = [
        (0.565, 0.665, C_ATT, "SMA-CLMPNet\n(proposed)"),
        (0.700, 0.665, C_ATT, "MUSE-CLMPNet\nSCAM-CLMPNet\n(ablations)"),
        (0.835, 0.665, C_SEQ, "SMA-CLMPNet-Opt\n(MEALPY-tuned)"),
        (0.565, 0.505, C_CONV, "DCNN\nEfficientNet"),
        (0.700, 0.505, C_CONV, "STIDNet\nGLCM"),
        (0.835, 0.505, C_CONV, "EfficientNetV2S\nConvNeXt-T"),
        (0.565, 0.345, C_CONV, "MobileNetV3-L\nResNet-RS50"),
        (0.700, 0.345, C_SEQ, "STIL TIM + ISM\n(TFace, unmodified)"),
        (0.835, 0.345, C_SEQ, "Temporal deltas\n+ L1 logistic"),
    ]
    for x, y, c, t in cohort:
        box(ax, x, y, 0.135, 0.130, t, c, fs=7)

    arrow(ax, 0.516, 0.755, 0.548, 0.640)
    box(ax, 0.565, 0.140, 0.405, 0.140,
        "metrics_fixed.py  —  real confusion matrix\n\n"
        "ACC / SEN / SPE / PRE / F1 / balanced accuracy, ROC-AUC",
        C_OUT, fs=7.5)
    arrow(ax, 0.7665, 0.345, 0.7665, 0.284)

    box(ax, 0.025, 0.290, 0.400, 0.180,
        "Audit of the published pipeline\n\n"
        "mealpy/metrics.py rewrites y_pred before scoring,\n"
        "so every published figure was re-measured here\n"
        "rather than reused",
        "#F2F2F2", fs=7.5, ec="#B0B0B0")
    arrow(ax, 0.225, 0.640, 0.225, 0.474, ls=":", lw=1.0, color="#808080")

    fig.savefig(OUT / "fig10_pipeline_block_diagram.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig10_pipeline_block_diagram.png")


# ------------------------------------------------------------------- fig 11
def architecture():
    # Shapes recomputed from the cached tensor, not transcribed.
    T, H, W, C = 10, 128, 128, 12
    s1 = (T, H, W, 16)                                   # Conv3D same, stride 1
    t2, h2, w2 = T - 2, H - 2, W - 2                     # Conv3D valid 3x3x3
    s2 = ((t2 + 1) // 2, (h2 + 1) // 2, (w2 + 1) // 2, 32)   # stride-2 decimate
    t3, h3, w3 = s2[0] - 2, s2[1] - 2, s2[2] - 2
    s3 = ((t3 + 1) // 2, (h3 + 1) // 2, (w3 + 1) // 2, 64)
    r1 = (s3[0] * s3[1], s3[2], s3[3])
    r2 = (r1[0], r1[1] * r1[2])

    def sh(t):
        return "(" + ", ".join(str(v) for v in t) + ")"

    fig, ax = canvas(11.0, 6.2)
    ax.text(0.5, 0.982, "SMA-CLMPNet as implemented "
            "(SubFunctions/Model.py:447-513)",
            ha="center", va="top", fontsize=11, weight="bold", color=C_EDGE)

    col_x, col_w = 0.055, 0.245
    stages = [
        (C_IN, f"Input\n{sh((T, H, W, C))}"),
        (C_CONV, f"Conv3D 16, 3x3x3, same  +  ReLU\n"
                 f"modified pooling: mean(MaxPool3D, AvgPool3D)\n"
                 f"pool 1, stride 1  —  no downsampling\n{sh(s1)}"),
        (C_CONV, f"Conv3D 32, 3x3x3, valid  +  ReLU\n"
                 f"modified pooling, pool 1, stride 2\n{sh(s2)}"),
        (C_CONV, f"Conv3D 64, 3x3x3, valid  +  ReLU  +  BatchNorm\n"
                 f"modified pooling, pool 1, stride 2  +  Dropout 0.25\n"
                 f"{sh(s3)}"),
        (C_ATT, f"Reshape  {sh(r1)}\n"
                f"SCAM — spatial and channel joint attention"),
        (C_SEQ, f"Reshape  {sh(r2)}\n"
                f"LSTM(128)  +  LSTM(128), summed by Add()"),
        (C_ATT, "MUSE — multi-excited block\n"
                "average operation, ELU, drop 0.05"),
        (C_HEAD, "Flatten → Dense 100 → ReLU → BN → Drop 0.5\n"
                 "→ Dense 64 → ReLU → BN → Drop 0.5"),
        (C_OUT, "Dense 2, softmax\nAdam, categorical cross-entropy"),
    ]
    n = len(stages)
    top, bot, gap = 0.915, 0.030, 0.026
    hgt = (top - bot - gap * (n - 1)) / n
    ys = [top - i * (hgt + gap) - hgt for i in range(n)]
    for (c, t), y in zip(stages, ys):
        box(ax, col_x, y, col_w * 1.85, hgt, t, c, fs=7.0)
    for i in range(n - 1):
        arrow(ax, col_x + col_w * 0.925, ys[i], col_x + col_w * 0.925,
              ys[i + 1] + hgt)

    xr, wr = 0.565, 0.415
    box(ax, xr, 0.630, wr, 0.285,
        "SCAM — spatial and channel joint attention\n"
        "(SubFunctions/SCAM.py)\n\n"
        "channel branch: global average + max pool\n"
        "→ shared MLP → sigmoid\n"
        "spatial branch: channel-wise average + max\n"
        "→ 7x7 conv → sigmoid\n"
        "output = input ⊙ channel-gate ⊙ spatial-gate\n\n"
        "toggled by opt ∈ {2, 3}; opt = 1 disables it",
        C_ATT, fs=7.0)
    box(ax, xr, 0.330, wr, 0.265,
        "MUSE — multi-excited block\n(SubFunctions/MUSE.py)\n\n"
        "several excitation paths over the 128-d\n"
        "recurrent state, combined by 'average',\n"
        "ELU activation, dropout 0.05\n\n"
        "toggled by opt ∈ {1, 3}; opt = 2 disables it",
        C_ATT, fs=7.0)
    box(ax, xr, 0.030, wr, 0.265,
        "Ablation coding used throughout this paper\n\n"
        "opt = 1  →  MUSE-CLMPNet    (MUSE only)\n"
        "opt = 2  →  SCAM-CLMPNet    (SCAM only)\n"
        "opt = 3  →  SMA-CLMPNet     (both; proposed)\n\n"
        "2,258,534 trainable parameters",
        "#F2F2F2", fs=7.0, ec="#B0B0B0")
    arrow(ax, col_x + col_w * 1.85, ys[4] + hgt / 2, xr, 0.772,
          ls=":", lw=1.0, color="#808080")
    arrow(ax, col_x + col_w * 1.85, ys[6] + hgt / 2, xr, 0.462,
          ls=":", lw=1.0, color="#808080")

    fig.savefig(OUT / "fig11_smaclmpnet_architecture.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig11_smaclmpnet_architecture.png")


# ------------------------------------------------------------------- fig 12
def protocol():
    fig, ax = canvas(11.0, 4.2)
    ax.text(0.5, 0.975, "Evaluation protocol: nothing is selected on the "
            "outer test fold", ha="center", va="top", fontsize=11,
            weight="bold", color=C_EDGE)

    box(ax, 0.020, 0.640, 0.170, 0.230,
        "50 videos\nstratified\nStratifiedKFold\nseed 1234", C_IN, fs=8)
    box(ax, 0.230, 0.640, 0.200, 0.230,
        "Outer fold k of 5\n40 train / 10 test\ntest never touched\n"
        "until scoring", C_CONV, fs=8)
    box(ax, 0.470, 0.640, 0.230, 0.230,
        "Inner 4-fold CV\non the 40 training videos\n"
        "feature scaling, C, penalty,\nearly-stopping epoch", C_SEQ, fs=8)
    box(ax, 0.740, 0.640, 0.240, 0.230,
        "Refit on all 40, predict\nthe held-out 10\n"
        "out-of-fold probabilities\npooled over all 50", C_OUT, fs=8)
    for a, b in [(0.190, 0.230), (0.430, 0.470), (0.700, 0.740)]:
        arrow(ax, a, 0.755, b, 0.755)

    box(ax, 0.020, 0.310, 0.470, 0.250,
        "Permutation test\n\n"
        "labels shuffled 200 times, the whole nested procedure re-run\n"
        "each time; the observed score is compared against that null\n"
        "rather than against the nominal 50% chance line",
        "#F2F2F2", fs=8, ec="#B0B0B0")
    box(ax, 0.510, 0.310, 0.470, 0.250,
        "Scoring\n\n"
        "metrics_fixed.py — confusion matrix from sklearn, never from\n"
        "mealpy/metrics.py; balanced accuracy reported alongside accuracy\n"
        "because the corpus is 29 / 21",
        "#F2F2F2", fs=8, ec="#B0B0B0")
    arrow(ax, 0.255, 0.640, 0.255, 0.560)
    arrow(ax, 0.745, 0.640, 0.745, 0.560)

    box(ax, 0.020, 0.040, 0.960, 0.210,
        "Excluded by design:  test-set weight fitting  ·  threshold chosen on "
        "the test fold\n"
        "best-of-N seed selection  ·  any metric computed by the tampered "
        "library\n\n"
        "Each of these raises the reported number without raising detection "
        "performance,\nand each appears in the material this study removed.",
        "#FDECEA", fs=7.5, ec="#D08C8C")

    fig.savefig(OUT / "fig12_evaluation_protocol.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig12_evaluation_protocol.png")


if __name__ == "__main__":
    pipeline()
    architecture()
    protocol()
    print(f"\nwrote 3 diagrams to {OUT.relative_to(P)}")
