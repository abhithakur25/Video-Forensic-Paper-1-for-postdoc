"""Build the results report from the saved arrays.

Analysis1/TP    - the reproduction run, scored by the pipeline as delivered.
                  Those scores are FABRICATED (see INTEGRITY_FINDING.md); they
                  are reported here only to quantify the gap.
Analysis1/TRUE  - the same models re-run and scored with a real confusion
                  matrix, plus the current-generation backbones and the
                  optimised SMA-CLMPNet.

Columns of the TRUE arrays: ACC, SEN, SPE, PRE, F1, BAL-ACC.
"""
import datetime
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path(r"C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1")
PUBLISHED = ["EfficientNet", "STIDNet", "DCNN", "GLCM", "BA-TFD",
             "MUSE-CLMPNet", "SCAM-CLMPNet", "SMA-CLMPNet"]
LATEST = ["EfficientNetV2S", "ConvNeXtTiny", "MobileNetV3Large", "ResNetRS50"]
PCTS = [40, 50, 60, 70, 80, 90]
METRICS = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-score",
           "Balanced acc."]
SESSION_START = datetime.datetime(2026, 8, 4, 12, 0, 0).timestamp()


def load_tp():
    d = P / "Analysis1" / "TP"
    f = d / "COM_A.npy"
    if not f.exists() or f.stat().st_mtime < SESSION_START:
        return None
    return {n: np.load(d / f"COM_{L}.npy")
            for L, n in zip("ABCDEFGH", PUBLISHED)}


def load_true():
    d = P / "Analysis1" / "TRUE"
    man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    out = {}
    for f in sorted(d.glob("*.npy")):
        out[f.stem] = np.load(f)
    return out, man


def table(rows, head):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    def line(r):
        return "| " + " | ".join(str(c).ljust(w[i])
                                 for i, c in enumerate(r)) + " |"
    return "\n".join([line(head),
                      "|" + "|".join("-" * (x + 2) for x in w)  + "|"]
                     + [line(r) for r in rows])


def mp(a):
    """Mean of a column, or an em dash when every entry is undefined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.nanmean(a) if not np.all(np.isnan(a)) else np.nan
    return "—" if np.isnan(v) else f"{v*100:.2f}"


def pc(v):
    return "—" if v is None or np.isnan(v) else f"{v*100:.2f}"


def main():
    tp = load_tp()
    tr, man = load_true()
    done = [int(round(p * 100)) for p in man["train_pcts"]]
    n_done = len(done)
    out = []

    out.append("# Results — Paper 1, corrected scoring\n")
    out.append(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S}.\n")
    out.append("> **The scores this project produces as delivered are "
               "fabricated.** The vendored `mealpy/metrics.py` discards model "
               "predictions and invents them. See "
               "[`INTEGRITY_FINDING.md`](INTEGRITY_FINDING.md). Everything "
               "below headed *measured* was scored with "
               "`Optimized/metrics_fixed.py`.\n")
    out.append(f"Splits completed in this run: "
               f"{', '.join(f'{d}%' for d in done)}. Baseline budget "
               f"{man['epochs_baseline'] if 'epochs_baseline' in man else 10} "
               f"epochs; SMA-CLMPNet-Opt {man['epochs']} epochs at batch "
               f"{man['batch_size']}.\n")

    # ------------------------------------------------- fabricated vs measured
    if tp is not None:
        out.append("\n## The gap: reported by the pipeline vs actually measured\n")
        out.append("Accuracy, mean over the splits evaluated in both runs. "
                   "The left column is what the delivered code prints; it is "
                   "not a measurement of anything.\n")
        rows = []
        for n in PUBLISHED:
            if n not in tr:
                continue
            fab = np.nanmean(tp[n][:n_done, 0])
            real = np.nanmean(tr[n][:, 0])
            bal = np.nanmean(tr[n][:, 5])
            rows.append([n, pc(fab), pc(real), pc(bal),
                         f"{(real-fab)*100:+.2f}"])
        if rows:
            out.append(table(rows, ["Model", "Fabricated", "Measured acc.",
                                    "Measured bal. acc.", "Δ (pp)"]))

    # ----------------------------------------------------- measured, by split
    out.append("\n\n## Measured accuracy by training percentage\n")
    rows = []
    for n in PUBLISHED + LATEST + ["SMA-CLMPNet-Opt"]:
        if n not in tr:
            continue
        a = tr[n][:, 0]
        cells = [pc(a[done.index(p)]) if p in done else "—" for p in PCTS]
        rows.append([n, *cells, mp(a), mp(tr[n][:, 5])])
    out.append(table(rows, ["Model"] + [f"{p}%" for p in PCTS]
                     + ["Mean acc.", "Mean bal. acc."]))
    out.append("\nChance is 50.00%. Majority-class-always is 58.00% accuracy "
               "and 50.00% balanced accuracy on this 29/21 corpus — any model "
               "at 50.00% balanced accuracy has learned nothing and is "
               "predicting one class for every input.\n")

    # ------------------------------------------------------ every metric, mean
    out.append("\n## Measured means, every metric\n")
    rows = []
    for n in PUBLISHED + LATEST + ["SMA-CLMPNet-Opt"]:
        if n not in tr:
            continue
        rows.append([n] + [mp(tr[n][:, i]) for i in range(6)])
    out.append(table(rows, ["Model"] + METRICS))

    # ------------------------------------------------------------- k-fold
    kd = P / "Analysis1" / "TRUE_KF"
    if (kd / "run_manifest.json").exists():
        kman = json.loads((kd / "run_manifest.json").read_text(encoding="utf-8"))
        kf = {f.stem: np.load(f) for f in sorted(kd.glob("*.npy"))}
        ks = kman["k_values"]
        out.append("\n\n## Measured k-fold comparison\n")
        out.append(f"Stratified k-fold, k = {', '.join(str(k) for k in ks)}, "
                   f"{kman['folds_per_k']} fold per k, scored with "
                   f"`metrics_fixed.py`. The published `KFAnalysis` could not "
                   f"be used: `Analysis.py:355` indexes `data['image']`, a key "
                   f"`ReadDataset` never stores.\n")
        rows = []
        for n in sorted(kf, key=lambda x: -np.nanmean(kf[x][:, 5])):
            a = kf[n]
            rows.append([n] + [mp(np.array([v])) for v in a[:, 0]]
                        + [mp(a[:, 0]), mp(a[:, 5])])
        out.append(table(rows, ["Model"] + [f"k={k}" for k in ks]
                         + ["Mean acc.", "Mean bal. acc."]))
        out.append("\nEach test fold holds 5-9 of the 50 videos, so one "
                   "misclassification moves accuracy by 11-20 pp. No "
                   "difference in this table is resolvable at that "
                   "granularity.\n")

    # -------------------------------------------------------- recipe ablation
    if "SMA-CLMPNet" in tr and "SMA-CLMPNet-Opt" in tr:
        base, new = tr["SMA-CLMPNet"], tr["SMA-CLMPNet-Opt"]
        out.append("\n\n## SMA-CLMPNet: published training recipe vs optimised\n")
        out.append("Identical architecture — the authors' MUSE block, SCAM "
                   "attention, 3D convolution stack and dual LSTM, rebuilt "
                   "layer for layer. Only the training recipe differs: batch "
                   "32→8 (44 training samples give 2 gradient steps per epoch "
                   "at batch 32), inputs standardised with training-split "
                   "statistics only, class weights for the 29/21 imbalance, "
                   "and a cosine-decayed learning rate over a longer budget.\n")
        rows = []
        for i, p in enumerate(done):
            rows.append([f"{p}%", pc(base[i, 0]), pc(new[i, 0]),
                         f"{(new[i,0]-base[i,0])*100:+.2f}",
                         pc(base[i, 5]), pc(new[i, 5]),
                         f"{(new[i,5]-base[i,5])*100:+.2f}"])
        rows.append(["**Mean**",
                     pc(np.nanmean(base[:, 0])), pc(np.nanmean(new[:, 0])),
                     f"{(np.nanmean(new[:,0])-np.nanmean(base[:,0]))*100:+.2f}",
                     pc(np.nanmean(base[:, 5])), pc(np.nanmean(new[:, 5])),
                     f"{(np.nanmean(new[:,5])-np.nanmean(base[:,5]))*100:+.2f}"])
        out.append(table(rows, ["Training %", "Base acc.", "Opt acc.", "Δ",
                                "Base bal.", "Opt bal.", "Δ"]))

    # ------------------------------------------- representation/model search
    v2 = P / "Optimized" / "optimize_v2.json"
    if v2.exists():
        j = json.loads(v2.read_text(encoding="utf-8"))
        out.append("\n\n## Model selection over richer representations\n")
        out.append(f"Protocol: {j['protocol']}. The deep models above are "
                   "trained on a single deterministic split; this section "
                   "instead does nested cross-validation, so the reported "
                   "score is estimated on folds the hyper-parameter selection "
                   "never saw. Representations preserve what a channel mean "
                   "destroys: multi-scale spatial layout, per-channel "
                   "distributions, and frame-to-frame temporal change.\n")
        rows = [[r["representation"], r["model"], f"{r['mean']*100:.2f}",
                 f"±{r['std']*100:.2f}"] for r in j["ranking"][:12]]
        out.append(table(rows, ["Representation", "Model",
                                "Nested bal. acc.", "SD across folds"]))
        w = j["winner"]
        out.append(f"\n**Permutation test on the winner** "
                   f"({w['model']} on {w['representation']}): observed "
                   f"{w['nested_bal_acc']*100:.2f}%, null mean "
                   f"{w['null_mean']*100:.2f}%, null 95th percentile "
                   f"**{w['null_p95']*100:.2f}%**, p = **{w['p_value']:.3f}**.\n")
        verdict = ("signal above chance" if w["p_value"] < 0.05 else
                   "**not distinguishable from chance** — the best honest "
                   "score does not beat what the same pipeline achieves on "
                   "randomly shuffled labels")
        out.append(f"Verdict: {verdict}.\n")

    fp = P / "Optimized" / "feature_probe.json"
    if fp.exists():
        j = json.loads(fp.read_text(encoding="utf-8"))
        pm = j["permutation"]
        out.append("\n## Independent probe\n")
        out.append("A separate, simpler probe (repeated stratified 5-fold, "
                   "logistic regression / RBF SVM / random forest on summary "
                   f"statistics) reached {pm['observed']*100:.2f}% balanced "
                   f"accuracy on '{pm['representation']}', against a null 95th "
                   f"percentile of {pm['null_p95']*100:.2f}% "
                   f"(p = {pm['p_value']:.3f}).\n")

    out.append("\n\n## Reading these numbers honestly\n")
    out.append("The corpus is 50 videos, 29 authentic / 21 forged, and the "
               "test partition ranges from 31 videos at the 40% split down to "
               "6 at the 90% split. One misclassification moves accuracy by "
               "3.23 pp at 40% and 16.67 pp at 90%. Differences smaller than "
               "that are noise, and nothing here supports a significance "
               "claim. Reporting a 90%-split number from six test videos is "
               "not meaningful regardless of how it is scored.\n")
    out.append("BA-TFD is absent throughout: its ViTDCNN definition applies "
               "`MaxPooling2D(1, 1)`, which does not downsample, so the "
               "flattened 1,048,576-element vector entering `Dense(2048)` "
               "requires an 8.6 GB weight matrix and exhausts memory at every "
               "batch size tested.\n")

    txt = "\n".join(out)
    (P / "Optimized" / "RESULTS.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
