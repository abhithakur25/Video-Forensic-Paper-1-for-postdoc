# Paper 1 — Genuine vs Fabricated Results

**Legend**

| Tag | Meaning |
|-----|---------|
| **GENUINE** | Measured on this machine with `Optimized/metrics_fixed.py` (sklearn). Real train/predict. |
| **FABRICATED / UNTRUSTWORTHY** | Comes from `mealpy.metrics.confusion_matrix` path (discards predictions, random label flips) or paper NPY regenerated via that path. **Do not use for scientific claims.** |
| **COLLAPSE†** | High Acc but Sen≈0 & Spec≈1 (or reverse) = majority/minority collapse — Acc is misleading even if GENUINE. |

**Data (all genuine runs):** Features.pkl proposed `(50,10,128,128,12)`, labels 29/21, majority baseline Acc = **58%**, test n≈10 @80% / n≈5 @90%.

---

## A. GENUINE results

### A1. Epochs = 2 (no class balance)

| Model | Train% | Acc | Sen | Spec | Prec | F1 | BalAcc | Note |
|-------|--------|-----|-----|------|------|-----|--------|------|
| DCNN | 80 | 80.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † collapse |
| DCNN | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| EfficientNetV2B0 | 80 | 60.0% | 0.0% | 75.0% | 0.0% | 0.0% | 37.5% | GENUINE |
| EfficientNetV2B0 | 90 | **80.0%** | 50.0% | 100% | 100% | **66.7%** | **75.0%** | GENUINE |
| MobileNetV2 | 80 | 70.0% | 50.0% | 75.0% | 33.3% | 40.0% | 62.5% | GENUINE |
| MobileNetV2 | 90 | 20.0% | 0.0% | 33.3% | 0.0% | 0.0% | 16.7% | GENUINE |
| STIDNet | 80 | 40.0% | 100% | 25.0% | 25.0% | 40.0% | 62.5% | GENUINE |
| STIDNet | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| P1-Proposed | 80 | 80.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| P1-Proposed | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |

Source: `evaluation_multi_ep2.csv`

---

### A2. Epochs = 20 (no class balance)

| Model | Train% | Acc | Sen | Spec | Prec | F1 | BalAcc | Note |
|-------|--------|-----|-----|------|------|-----|--------|------|
| DCNN | 80 | 60.0% | 0.0% | 75.0% | 0.0% | 0.0% | 37.5% | GENUINE |
| DCNN | 90 | **80.0%** | 50.0% | 100% | 100% | **66.7%** | **75.0%** | GENUINE |
| EfficientNetV2B0 | 80 | 50.0% | 50.0% | 50.0% | 20.0% | 28.6% | 50.0% | GENUINE |
| EfficientNetV2B0 | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| MobileNetV2 | 80 | 40.0% | 0.0% | 50.0% | 0.0% | 0.0% | 25.0% | GENUINE |
| MobileNetV2 | 90 | 60.0% | 50.0% | 66.7% | 50.0% | 50.0% | 58.3% | GENUINE |
| STIDNet | 80 | **70.0%** | 100% | 62.5% | 40.0% | **57.1%** | **81.3%** | GENUINE best BalAcc |
| STIDNet | 90 | 40.0% | 0.0% | 66.7% | 0.0% | 0.0% | 33.3% | GENUINE |
| P1-Proposed | 80 | 20.0% | 100% | 0.0% | 20.0% | 33.3% | 50.0% | GENUINE |
| P1-Proposed | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |

Source: `evaluation_multi_ep20.csv`

---

### A3. Epochs = 50

| Model | Train% | Acc | Sen | Spec | Prec | F1 | BalAcc | Note |
|-------|--------|-----|-----|------|------|-----|--------|------|
| DCNN | 80 | 40.0% | 50.0% | 37.5% | 16.7% | 25.0% | 43.8% | GENUINE |
| DCNN | 90 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | GENUINE |
| EfficientNetV2B0 | 80 | **80.0%** | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † collapse |
| EfficientNetV2B0 | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| MobileNetV2 | 80 | 50.0% | 50.0% | 50.0% | 20.0% | 28.6% | 50.0% | GENUINE |
| MobileNetV2 | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| STIDNet | 80 | 50.0% | 50.0% | 50.0% | 20.0% | 28.6% | 50.0% | GENUINE |
| STIDNet | 90 | 40.0% | 50.0% | 33.3% | 33.3% | 40.0% | 41.7% | GENUINE |
| P1-Proposed | 80 | 20.0% | 100% | 0.0% | 20.0% | 33.3% | 50.0% | GENUINE |
| P1-Proposed | 90 | 40.0% | 100% | 0.0% | 40.0% | **57.1%** | 50.0% | GENUINE |

Source: `evaluation_multi_ep50.csv`

---

### A4. Epochs = 100

| Model | Train% | Acc | Sen | Spec | Prec | F1 | BalAcc | Note |
|-------|--------|-----|-----|------|------|-----|--------|------|
| DCNN | 80 | 50.0% | 50.0% | 50.0% | 20.0% | 28.6% | 50.0% | GENUINE |
| DCNN | 90 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | GENUINE |
| EfficientNetV2B0 | 80 | **80.0%** | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| EfficientNetV2B0 | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| MobileNetV2 | 80 | 60.0% | 100% | 50.0% | 33.3% | **50.0%** | **75.0%** | GENUINE |
| MobileNetV2 | 90 | 20.0% | 0.0% | 33.3% | 0.0% | 0.0% | 16.7% | GENUINE |
| STIDNet | 80 | 60.0% | 50.0% | 62.5% | 25.0% | 33.3% | 56.3% | GENUINE |
| STIDNet | 90 | 20.0% | 0.0% | 33.3% | 0.0% | 0.0% | 16.7% | GENUINE |
| P1-Proposed | 80 | **80.0%** | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| P1-Proposed | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |

Source: `evaluation_multi_ep100.csv`

---

### A5. Epochs = 20 + class_weight + oversample (GENUINE + balanced training)

| Model | Train% | Acc | Sen | Spec | Prec | F1 | BalAcc | Note |
|-------|--------|-----|-----|------|------|-----|--------|------|
| DCNN | 80 | 30.0% | 50.0% | 25.0% | 14.3% | 22.2% | 37.5% | GENUINE |
| DCNN | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |
| EfficientNetV2B0 | 80 | 50.0% | 50.0% | 50.0% | 20.0% | 28.6% | 50.0% | GENUINE |
| EfficientNetV2B0 | 90 | 60.0% | 50.0% | 66.7% | 50.0% | **50.0%** | 58.3% | GENUINE |
| MobileNetV2 | 80 | **70.0%** | **100%** | 62.5% | 40.0% | **57.1%** | **81.3%** | GENUINE best balanced |
| MobileNetV2 | 90 | 40.0% | 50.0% | 33.3% | 33.3% | 40.0% | 41.7% | GENUINE |
| STIDNet | 80 | **80.0%** | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † Acc only |
| STIDNet | 90 | 20.0% | 50.0% | 0.0% | 25.0% | 33.3% | 25.0% | GENUINE |
| P1-Proposed | 80 | 20.0% | 100% | 0.0% | 20.0% | 33.3% | 50.0% | GENUINE |
| P1-Proposed | 90 | 60.0% | 0.0% | 100% | 0.0% | 0.0% | 50.0% | GENUINE † |

Source: `evaluation_multi_ep20_bal_cw_os.csv`

---

### A6. Weight-scale tune best configs (GENUINE sweep means @80% train)

| Model | minority_scale | oversample_ratio | Mean BalAcc | Mean Acc | Tag |
|-------|----------------|------------------|-------------|----------|-----|
| DCNN | 3.0 | 1.5 | 68.8% | 50.0% | GENUINE |
| EfficientNetV2B0 | 1.0 | 1.5 | **81.2%** | 70.0% | GENUINE |
| MobileNetV2 | 2.0 | 1.0 | **81.2%** | 70.0% | GENUINE |
| STIDNet | 3.0 | 1.5 | 68.8% | 80.0% | GENUINE (Acc may collapse) |
| P1-Proposed | 2.0 | 1.0 | 50.0% | 80.0% | GENUINE † Acc collapse likely |
| RF | 1.0 | 1.0 | 62.5% | 70.0% | GENUINE |
| GBM | 1.0 | 1.5 | 43.8% | 70.0% | GENUINE |

Source: `weight_tune_best_summary.txt` / `logs/weight_tune.log`

---

### A7. Best GENUINE scores (summary)

| Metric | Best genuine value | Model / setting |
|--------|--------------------|-----------------|
| Highest Acc (non-collapse preferred) | **80%** | DCNN ep20@90%; EffNet ep2@90% (with F1 0.67) |
| Highest BalAcc | **81.3%** | STIDNet ep20@80%; MobileNet bal ep20@80% |
| Highest F1 | **66.7%** | EffNet ep2@90%; DCNN ep20@90% |
| Highest Sen with decent BalAcc | **100%** Sen, 81.3% Bal | STIDNet ep20; MobileNet bal ep20 |

**Target 95–99% Acc: NOT achieved with any GENUINE run.**

---

## B. FABRICATED / UNTRUSTWORTHY results (tagged)

### B1. Paper artefact `Analysis/TP/COM_A.npy`

**TAG: FABRICATED / UNTRUSTWORTHY** — regenerated via original pipeline that uses `mealpy.metrics.confusion_matrix` (random flips; does not use model predictions).

| Train % | Acc | Sen | Spec | Prec | F1 | Tag |
|---------|-----|-----|------|------|-----|-----|
| 40% | 94.42% | 93.67% | 94.79% | 95.17% | 94.41% | **FABRICATED** |
| 50% | 93.28% | 91.18% | 94.31% | 95.39% | 93.24% | **FABRICATED** |
| 60% | 96.06% | 93.93% | 97.11% | 98.20% | 96.02% | **FABRICATED** |
| 70% | 93.58% | 95.65% | 92.52% | 91.51% | 93.53% | **FABRICATED** |
| 80% | 94.27% | 97.47% | 92.61% | 91.06% | 94.16% | **FABRICATED** |
| 90% | 93.34% | 91.52% | 94.23% | 95.15% | 93.30% | **FABRICATED** |

Any bar/line charts under `Results/TP/` and `Results/KF/` that were produced from `Analysis/*.npy` via `Main.py` / original Evaluate path: **FABRICATED / UNTRUSTWORTHY** for the same reason.

### B2. Why they are tagged fabricated

From `Optimized/INTEGRITY_FINDING.md` and `mealpy/metrics.py`:

- `_check_targets()` **ignores** the model’s `y_pred`.
- Rebuilds labels from class counts and **randomly flips** a fraction `per ~ U(0.09, 0.45)`.
- Therefore Acc/Sen/Spec/Prec/F1 are **RNG draws**, not model performance.
- Repeated calls on the same model disagree; a perfect predictor still scores ~0.65–1.00 randomly.

### B3. Other sources to treat as FABRICATED if scored via Evaluate.py

| Source | Tag |
|--------|-----|
| `SubFunctions/Evaluate.py` → `mealpy.metrics.confusion_matrix` | **FABRICATED** |
| `Analysis/**/*.npy`, `Analysis1/**/*.npy` (paper COM/PERF) | **FABRICATED** (unless re-measured with metrics_fixed) |
| `Results/TP/*`, `Results/KF/*` plots from those NPYs | **FABRICATED** |
| Claims of ~93–98% from original Paper 1 full analysis UI path | **FABRICATED** until re-run with metrics_fixed |

---

## C. Quick rule

| Use for publication / postdoc claims? | Which results |
|--------------------------------------|---------------|
| **YES** | Section A only (`Optimized/results/evaluation_multi_*.csv`) |
| **NO** | Section B (`Analysis/TP/COM_A.npy`, mealpy Evaluate path, related plots) |
| **Caution** | GENUINE rows marked **† collapse** — report BalAcc/F1, not Acc alone |

Files on disk: this document is  
`Optimized/results/GENUINE_vs_FABRICATED_RESULTS.md`
