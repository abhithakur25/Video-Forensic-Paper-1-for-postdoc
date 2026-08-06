# Provenance: what this repository contains, and what was removed

**Every result in this repository is a measurement.** Fabricated results were
removed on 2026-08-06 and are no longer tracked here.

There were exactly two classes of result, and nothing in between.

| Tag | Meaning |
|---|---|
| 🟢 **GENUINE** | Scored by `Optimized/metrics_fixed.py` — a real `sklearn` confusion matrix over the model's actual predictions. Reproducible and deterministic: re-running the k-fold reproduced EfficientNet 55.56 / 50.00 and STIDNet 66.67 / 67.50 to the digit. |
| 🔴 **FABRICATED** | Not a measurement of anything. **Removed from the repository.** |

Removed artifacts were **moved to `../_FABRICATED_QUARANTINE_Paper1/`**, one
level above the repository, not unlinked — they are the evidence for the
integrity finding and may be needed to substantiate it. They are outside the
working tree and outside git. Reproduce the removal with
`python Optimized/purge_fabricated.py` (dry run) / `--apply`.

---

## What was removed, and on what grounds

161 files, 26.90 MB. Three independent grounds, all verified.

### 1. Scored through the tampered metric

`mealpy/metrics.py:16-75`, inside `_check_targets()`, discards the model's
predictions:

```python
y_pred = y_true.copy()                                # predictions discarded
va = random.sample(range(1, len(y_true)), int(len(y_true) * per))
for i in va:
    y_pred[i] = (random.sample(range(0, n), 1))[0]    # randomise a fraction
```

Expected accuracy is `1 - per/2` with `per ~ U(0.065, 0.452)`, i.e. uniform on
**0.77–0.955** — exactly the band reported. A perfect predictor scores
0.645–1.000 through this function, and two identical calls disagree.

| Removed | What it was |
|---|---|
| `Analysis/` (28 files) | The authors' own arrays, dated 2025-03-19. The source of every metric figure in the manuscript. |
| `Analysis1/TP/`, `Analysis1/KF/` (26) | Re-runs through the same scorer. |
| `Analysis1/TPR.npy`, `FPR.npy` | ROC points from the invented vector. |
| `Analysis1/TRUE_LATEST/` (5) | The run that **exposed** the tamper — four unrelated backbones returning byte-identical scores. |
| `logs/evaluation_tp_sweep.log`, `evaluation_kfold.log`, `evaluation_kfold_aborted.log` | Console records of the above. |

### 2. Plotted from (1)

| Removed | What it was |
|---|---|
| `Results/TP/` (45), `Results/KF/` (45) | Comparative and performance bar/line figures. 24.3 MB. |
| `Results/RocAnalysis/` (2) | ROC figures. |
| `Results/Results.xlsx` | The manuscript's TP and KF metric tables. |

### 3. Describes a corpus that does not exist

`Features/Features.pkl` holds **50 videos, 29 authentic / 21 forged**. These
artifacts describe something else entirely, so they cannot have been produced
from the shipped data by *any* scorer, tampered or not:

| Removed | Why it is impossible |
|---|---|
| `Results/Class.png` | Claims **1000 Normal / 1000 Scam**. The corpus is 29/21. |
| `Results/ConfusionMatrix.png` | Totals **400 test samples at 200/200**, accuracy 97.25%. The entire corpus is 50 videos, and the largest test split is 31. |
| `Results/Features.csv` | Ablation accuracies 95.58–97.92, quoted to **12 significant figures** (`97.0154535352`) on a 50-video corpus where one video is worth 2 percentage points. |
| `Results/Features.jpg` | Bar chart of `Features.csv`. |

This third group is a separate finding from the tampered metric. Those numbers
were not computed at all.

---

## What was kept

### 🟢 Measured results

| Path | What it is |
|---|---|
| `Analysis1/TRUE/` + `run_manifest.json` | Training-percentage sweep, 12 models × 6 splits, real confusion matrix. |
| `Analysis1/TRUE_KF/` + `run_manifest.json` | K-fold, k = 6…10, real confusion matrix. |
| `Optimized/optimize_v2.json` | 14 representations × 9 model families, nested CV, 100-shuffle permutation test. |
| `Optimized/optimize_v3.json` | Higher-order temporal features and ensembles. |
| `Optimized/optimize_weights.json` | 30-config class-weight / calibration / threshold sweep, all selected **inside** training folds. |
| `Optimized/feature_probe.json` | Independent simpler probe. |
| `Optimized/paper2_model.json` | Paper 2's BiLSTMGBM at its own settings, **test-set fitting omitted**. |
| `Optimized/final_tables.json` | Metric tables for the best honest pipeline. |
| `Optimized/metrics_fixed.py` | The corrected scorer. Self-test: perfect → 1.0000, inverted → 0.0000, all-zeros → 0.4839, stable over 50 calls. |

### Genuine non-metric artifacts

| Path | What it is |
|---|---|
| `Results/ImageResults/` (1,300 files) | Real GradCAM, LDZP, optical-flow, ResNet-statistic image outputs. No metric involved. |
| `Results/Arc.png` | Architecture diagram. |
| `Features/Features.pkl` | The actual 50-video feature set. |

### Source code left in place

`mealpy/metrics.py` — **the tampered file itself is still present.** It is
library source, not a result: `SubFunctions/Evaluate.py` imports it, and
deleting it breaks the codebase's own imports. It is documented in full in
[`INTEGRITY_FINDING.md`](INTEGRITY_FINDING.md), and nothing in this repository
is scored by it any more. Anything run through `SubFunctions/Evaluate.py`
still produces fabricated numbers; the pipeline of record is
`Optimized/optimize_models.py`, which uses `metrics_fixed.py`.

Consequence of the removal: `Main.py` and `driver.py plots` no longer run —
they read `Analysis/*.npy`. That is intended. Those figures were the
fabricated ones.

---

## The genuine headline numbers

Corpus: 50 videos, 29 authentic / 21 forged, one feature tensor per video.
Chance = 50.00% balanced accuracy. Majority-class-always = 58.00% accuracy but
50.00% balanced accuracy.

**Best honest result: 77.17% balanced accuracy** — L1 logistic regression on
frame-to-frame temporal delta statistics, nested CV, permutation p = 0.0099
(null mean 50.35%, null 95th percentile 63.51%). Real signal, far below the
manuscript's claims.

Everything else — 12 deep models, 4 current-generation backbones, 19
representations, 3 ensemble schemes, Paper 2's architecture at 500 epochs —
lands between 45% and 62% balanced accuracy.

## Not measured

| Path | Status |
|---|---|
| `FFPP/` | Pipeline built and smoke-tested. **Never run** — `DATASET/` has the directory tree but no videos. No FF++ number here is a measurement. |
| Literature figures in `COMPARISON.md` | **Quoted from published papers, not reproduced.** Context only. |
