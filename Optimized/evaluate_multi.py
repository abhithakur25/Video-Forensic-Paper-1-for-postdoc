# -*- coding: utf-8 -*-
"""Paper 1 multi-model evaluation (Paper-2 style).

Uses honest sklearn metrics (metrics_fixed), NOT mealpy.metrics.
Saves tables, CSV, NPY, and comparison figures under Optimized/results/.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "Optimized" / "results"
FIGS = PROJECT / "Optimized" / "figures"
LOGS = PROJECT / "Optimized" / "logs"


def log(msg):
    line = f"[P1-eval] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "evaluate_multi.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_features():
    import pickle

    path = PROJECT / "Features" / "Features.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def plot_accuracy(grid, models, pcts, path):
    # grid[model][split] = [acc,sen,spe,pre,f1,bal]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    w = 0.35
    for i, tp in enumerate(pcts):
        vals = []
        for m in models:
            row = grid[m][i]
            vals.append(row[0] * 100 if row[0] == row[0] else 0)
        ax.bar(x + (i - 0.5) * w, vals, w, label=f"Train {int(tp*100)}%")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metrics_bar(grid, models, split_i, pct, path):
    metric_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_names))
    w = 0.15
    colors = ["#4a7ab5", "#e67e22", "#3cb371", "#9b59b6", "#e74c3c"]
    for i, m in enumerate(models):
        row = grid[m][split_i]
        vals = [(row[j] * 100 if row[j] == row[j] else 0) for j in range(5)]
        ax.bar(x + (i - len(models) / 2) * w + w / 2, vals, w, label=m, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ranking(grid, models, split_i, path):
    vals = []
    for m in models:
        row = grid[m][split_i]
        vals.append(row[0] * 100 if row[0] == row[0] else 0)
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = [models[i] for i in order]
    v = [vals[i] for i in order]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(y)))
    ax.barh(y, v, color=colors)
    for yi, vi in enumerate(v):
        ax.annotate(f"{vi:.1f}%", xy=(vi, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-pcts", default="0.8,0.9")
    ap.add_argument(
        "--models",
        default="DCNN,EfficientNetV2B0,MobileNetV2,STIDNet,P1-Proposed",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-weight", action="store_true",
                    help="balanced class_weight in model.fit")
    ap.add_argument("--oversample", action="store_true",
                    help="random oversample minority class on train set only")
    ap.add_argument("--tag", default="",
                    help="extra filename tag e.g. bal for balanced runs")
    args = ap.parse_args()

    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    from Optimized.feature_adapters import train_test_split_arrays
    from Optimized.metrics_fixed import evaluation_metrics, majority_baseline_accuracy
    from Optimized.MultiModel import (
        MODEL_REGISTRY, LATEST_BACKBONE, LATEST_BACKBONE_REASON, run_model,
    )

    epochs = int(args.epochs)
    pcts = [float(x) for x in args.train_pcts.split(",")]
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in wanted:
        if m not in MODEL_REGISTRY:
            log(f"unknown model {m}; known={list(MODEL_REGISTRY)}")
            return 1

    data = load_features()
    labels = np.asarray(data["labels"]).ravel()
    proposed = np.asarray(data["proposed"])
    u, c = np.unique(labels, return_counts=True)
    maj = majority_baseline_accuracy(labels)
    log(f"proposed {proposed.shape} labels {dict(zip(u.tolist(), c.tolist()))}")
    log(f"majority baseline accuracy = {maj:.4f} ({maj*100:.1f}%)")
    log(f"LATEST={LATEST_BACKBONE} — {LATEST_BACKBONE_REASON}")
    use_cw = bool(args.class_weight)
    use_os = bool(args.oversample)
    tag = (args.tag or "").strip()
    if use_cw and "cw" not in tag:
        tag = (tag + "_cw").strip("_")
    if use_os and "os" not in tag:
        tag = (tag + "_os").strip("_")
    log(f"epochs={epochs} models={wanted} train_pcts={pcts}")
    log(f"balance: class_weight={use_cw} oversample={use_os} tag={tag!r}")

    import tensorflow as tf
    from tensorflow import keras

    log(f"tf={tf.__version__} keras={keras.__version__}")

    grid = {m: [] for m in wanted}
    t0 = time.time()

    for si, tp in enumerate(pcts):
        # split on proposed only; same indices for all models
        # rebuild full dict split to keep seed consistent via labels alone
        x_tr, x_te, y_tr, y_te = train_test_split_arrays(
            proposed, labels, train_size=tp, seed=args.seed + si
        )
        log(f"===== split {si+1}/{len(pcts)} TP={tp} train={len(y_tr)} test={len(y_te)} "
            f"y_tr={dict(zip(*np.unique(y_tr, return_counts=True)))} "
            f"y_te={dict(zip(*np.unique(y_te, return_counts=True)))} =====")
        for name in wanted:
            t = time.time()
            try:
                pred = run_model(
                    name, x_tr, y_tr, x_te, y_te, epochs=epochs,
                    use_class_weight=use_cw, use_oversample=use_os,
                    seed=args.seed + si,
                )
                pred = np.asarray(pred).reshape(-1)
                if len(pred) != len(y_te):
                    raise RuntimeError(f"pred len {len(pred)} != {len(y_te)}")
                mets = evaluation_metrics(y_te, pred)
                grid[name].append(mets)
                log(
                    f"  {name:<16} ACC={mets[0]:.4f} SEN={mets[1]:.4f} SPE={mets[2]:.4f} "
                    f"PRE={mets[3]:.4f} F1={mets[4]:.4f} BAL={mets[5]:.4f}  ({time.time()-t:.0f}s)"
                )
            except Exception as e:
                grid[name].append([float("nan")] * 6)
                log(f"  {name:<16} FAILED {type(e).__name__}: {e}")
                traceback.print_exc()

    # save npy
    letter = {m: chr(ord("A") + i) for i, m in enumerate(wanted)}
    for name in wanted:
        arr = np.asarray(grid[name], dtype=float)
        np.save(OUT / f"MULTI_{letter[name]}_{name.replace('/', '_')}.npy", arr)

    # text report
    metric_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "BalAcc"]
    lines = [
        "Paper 1 multi-model evaluation (honest sklearn metrics on Features.pkl)",
        f"proposed features : {proposed.shape}",
        f"samples  : {len(labels)}  balance {dict(zip(u.tolist(), c.tolist()))}",
        f"majority baseline ACC : {maj:.4f}",
        f"epochs   : {epochs}",
        f"train_pcts: {pcts}",
        f"models   : {wanted}",
        f"class_weight : {use_cw}",
        f"oversample   : {use_os}",
        f"tag      : {tag or '(none)'}",
        f"latest   : {LATEST_BACKBONE} — {LATEST_BACKBONE_REASON}",
        f"tf/keras : {tf.__version__} / {keras.__version__}",
        f"elapsed  : {time.time()-t0:.0f}s",
        f"metric_path: Optimized/metrics_fixed.py (NOT mealpy.metrics)",
        "",
        "NOTE: Original SubFunctions/Evaluate.py routes through a tampered",
        "mealpy.metrics.confusion_matrix (see Optimized/INTEGRITY_FINDING.md).",
        "All numbers below are re-measured with sklearn.",
        "",
    ]
    for mi, metric in enumerate(metric_names):
        hdr = f"{metric:<14}" + "".join(f"{int(p*100):>10}%" for p in pcts)
        lines += [hdr, "-" * len(hdr)]
        for name in wanted:
            cells = []
            for i in range(len(pcts)):
                v = grid[name][i][mi]
                cells.append(f"{v:>11.4f}" if v == v else f"{'FAILED':>11}")
            lines.append(f"{name:<14}" + "".join(cells))
        lines.append("")

    lines.append("Per-model status:")
    for name in wanted:
        fails = sum(1 for row in grid[name] if any(v != v for v in row))
        ok = len(pcts) - fails
        lines.append(f"  {name}: {ok}/{len(pcts)} splits OK"
                     + (f", {fails} FAILED" if fails else ""))

    lines += [
        "",
        "GENUINE ONLY: scores from Optimized/metrics_fixed.py (sklearn).",
        "Fabricated Analysis/COM_A and mealpy Evaluate paths are not used.",
        "",
    ]

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    suffix = f"ep{epochs}" + (f"_{tag}" if tag else "")
    out_txt = f"evaluation_multi_{suffix}.txt"
    (OUT / out_txt).write_text(text, encoding="utf-8")
    (LOGS / out_txt).write_text(text, encoding="utf-8")

    csv_path = OUT / f"evaluation_multi_{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "train_pct", "ACC", "SEN", "SPE", "PRE", "F1", "BAL_ACC", "status"])
        for name in wanted:
            for i, tp in enumerate(pcts):
                row = grid[name][i]
                ok = all(v == v for v in row)
                w.writerow([
                    name, int(tp * 100),
                    *[f"{v:.6f}" if v == v else "" for v in row],
                    "OK" if ok else "FAILED",
                ])
    log(f"wrote {csv_path}")

    # figures (no titles on image)
    try:
        plot_accuracy(grid, wanted, pcts, FIGS / "Fig1_machine_accuracy.png")
        for i, tp in enumerate(pcts):
            plot_metrics_bar(grid, wanted, i, tp, FIGS / f"Fig_metrics_{int(tp*100)}.png")
        plot_ranking(grid, wanted, len(pcts) - 1, FIGS / "Fig_ranking_last_split.png")
        log(f"figures -> {FIGS}")
    except Exception as e:
        log(f"figure generation failed: {e}")

    ok_models = [m for m in wanted if any(all(v == v for v in row) for row in grid[m])]
    log(f"OK models: {ok_models}")
    return 0 if len(ok_models) >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
