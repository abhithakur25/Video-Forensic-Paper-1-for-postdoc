# Paper 1 — Multi-model evaluation summary (machine, honest metrics)

**Date:** 2026-08-05  
**Project:** `CODE_28-04-2025_Paper1`  
**Features:** `Features/Features.pkl` → `proposed` tensor `(50, 10, 128, 128, 12)`  
**Labels:** 29 authentic (0), 21 forged (1) — majority baseline **58.0%** accuracy  
**Metrics path:** `Optimized/metrics_fixed.py` (sklearn) — **not** tampered `mealpy.metrics`

---

## Run E — epochs = 20 + class_weight + oversample (requested)

**Protocol:** epochs=20, `--class-weight --oversample`, wall ~558 s  
**Files:** `results/evaluation_multi_ep20_bal_cw_os.txt`, `.csv`

| Model | Acc @80% | Acc @90% | BalAcc @80% | BalAcc @90% | F1 @80% | F1 @90% |
|-------|----------|----------|-------------|-------------|---------|---------|
| DCNN | 30% | 60% | 37.5% | 50% | 0.22 | 0.00 |
| EfficientNetV2B0 | 50% | 60% | 50% | 58.3% | 0.29 | **0.50** |
| MobileNetV2 | **70%** | 40% | **81.3%** | 41.7% | **0.57** | 0.40 |
| STIDNet | 80%† | 20% | 50% | 25% | 0.00 | 0.33 |
| P1-Proposed | 20% | 60% | 50% | 50% | 0.33 | 0.00 |

† Majority collapse. **Best balanced result this run:** MobileNetV2 @80% (BalAcc 81.3%, F1 0.57, Sen=1.0).

vs plain epochs=20 (no balance): STIDNet had BalAcc 81% / F1 0.57 @80%; balancing shifts the win to MobileNetV2 and raises EffNet F1@90% to 0.50.

---

## Run D — epochs = 100 (requested re-run)

**Protocol:** epochs=100, train 80% / 90%, wall ~1060 s  
**Files:** `results/evaluation_multi_ep100.txt`, `evaluation_multi_ep100.csv`

| Model | Acc @80% | Acc @90% | BalAcc @80% | BalAcc @90% | F1 @80% | F1 @90% |
|-------|----------|----------|-------------|-------------|---------|---------|
| DCNN | 50% | 0% | 50% | 0% | 0.29 | 0.00 |
| EfficientNetV2B0 | **80%**† | 60% | 50% | 50% | 0.00 | 0.00 |
| MobileNetV2 | 60% | 20% | **75%** | 17% | **0.50** | 0.00 |
| STIDNet | 60% | 20% | 56% | 17% | 0.33 | 0.00 |
| P1-Proposed | **80%**† | 60% | 50% | 50% | 0.00 | 0.00 |

† Majority collapse (Sen=0, Spec=1). **More epochs alone does not fix N=50 imbalance.**

---

## Run C — epochs = 50 (requested re-run)

**Protocol:** epochs=50, train 80% / 90%, wall ~484 s  
**Files:** `results/evaluation_multi_ep50.txt`, `evaluation_multi_ep50.csv`

### Accuracy (epochs=50)

| Model | Acc @80% | Acc @90% |
|-------|----------|----------|
| DCNN | 40.00% | 0.00% |
| EfficientNetV2B0 | **80.00%** | 60.00% |
| MobileNetV2 | 50.00% | 60.00% |
| STIDNet | 50.00% | 40.00% |
| P1-Proposed | 20.00% | 40.00% |

### Balanced accuracy / F1 (epochs=50)

| Model | BalAcc @80% | BalAcc @90% | F1 @80% | F1 @90% |
|-------|-------------|-------------|---------|---------|
| DCNN | 43.75% | 0.00% | 0.25 | 0.00 |
| EfficientNetV2B0 | 50.00% | 50.00% | 0.00 | 0.00 |
| MobileNetV2 | 50.00% | 50.00% | 0.29 | 0.00 |
| STIDNet | 50.00% | 41.67% | 0.29 | 0.40 |
| P1-Proposed | 50.00% | 50.00% | 0.33 | **0.57** |

**Note:** EfficientNetV2B0 Acc@80%=80% is majority collapse (Sen=0, Spec=1). P1-Proposed F1@90% improves to 0.57 but Acc remains low.

---

## Run B — epochs = 20

**Protocol:** epochs=20, train 80% / 90%, wall ~232 s  
**Files:** `results/evaluation_multi_ep20.txt`, `evaluation_multi_ep20.csv`

### Accuracy (epochs=20)

| Model | Acc @80% | Acc @90% |
|-------|----------|----------|
| DCNN | 60.00% | **80.00%** |
| EfficientNetV2B0 | 50.00% | 60.00% |
| MobileNetV2 | 40.00% | 60.00% |
| STIDNet | **70.00%** | 40.00% |
| P1-Proposed | 20.00% | 60.00% |

### Balanced accuracy (epochs=20)

| Model | BalAcc @80% | BalAcc @90% |
|-------|-------------|-------------|
| DCNN | 37.50% | **75.00%** |
| EfficientNetV2B0 | 50.00% | 50.00% |
| MobileNetV2 | 25.00% | 58.33% |
| STIDNet | **81.25%** | 33.33% |
| P1-Proposed | 50.00% | 50.00% |

### Best F1 (epochs=20)

| Model | F1 @80% | F1 @90% |
|-------|---------|---------|
| DCNN | 0.00 | **0.67** |
| EfficientNetV2B0 | 0.29 | 0.00 |
| MobileNetV2 | 0.00 | 0.50 |
| STIDNet | **0.57** | 0.00 |
| P1-Proposed | 0.33 | 0.00 |

---

## Run A — epochs = 2 (baseline)

| Model | Acc @80% | Acc @90% | BalAcc @80% | BalAcc @90% |
|-------|----------|----------|-------------|-------------|
| DCNN | 80% | 60% | 50% | 50% |
| EfficientNetV2B0 | 60% | 80% | 37.5% | **75%** |
| MobileNetV2 | 70% | 20% | **62.5%** | 16.7% |
| STIDNet | 40% | 60% | **62.5%** | 50% |
| P1-Proposed | 80% | 60% | 50% | 50% |

## Interpretation

- Test sets remain tiny (n=10 @80%, n=5 @90%) → high variance between runs.
- More epochs **does not guarantee higher Acc**; STIDNet BalAcc@80% improved strongly (62.5%→81.25%), DCNN Acc@90% rose (60%→80%).
- P1-Proposed still unstable (majority collapse on some splits).
- Paper artefact `Analysis/TP/COM_A.npy` ~93% ACC is reference only.

## Package layout

```
Optimized/
  INTEGRITY_FINDING.md
  metrics_fixed.py
  feature_adapters.py
  MultiModel.py
  evaluate_multi.py
  results/          # CSV, TXT, NPY
  figures/          # comparison charts (no on-image titles)
  working_code/     # original Model/MUSE/SCAM + optimized copies
  logs/
```

Re-run:
```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd C:\Users\USER\Downloads\PostDoc\CODE_28-04-2025_Paper1
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 5 --train-pcts "0.7,0.8,0.9"
```
