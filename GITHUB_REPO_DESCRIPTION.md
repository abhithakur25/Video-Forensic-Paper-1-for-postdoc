# Video Forensic Paper 1 for Postdoc

**Private research repository** for Paper 1: *Design and development of video forgery model using deep learning with attention mechanisms* (LDZP + optical flow + 3D-CNN/LSTM/MUSE/SCAM).

Aligned with the RIITC–NKUST postdoctoral programme (semantic media forensics) and the Paper 2 multi-model evaluation workflow.

---

## What this repo contains

| Area | Contents |
|------|----------|
| **Core app** | `GUI.py`, `Main.py`, `SubFunctions/` (features, GradCAM, LDZP, MUSE, SCAM, Model) |
| **Optimized multi-model package** | `Optimized/` — honest metrics, multi-model training, class-weight tuning, full result report |
| **Results** | `Optimized/results/*.csv|txt|npy`, comparison figures, Word report |
| **Paper artefacts** | Analysis NPY (reference), ROC/TP/KF plots (small), Word paper draft |
| **Driver** | `.claude/skills/run-video-forgery-paper1/driver.py` for headless GUI/plots/check |

**Not in Git** (too large): `Features/Features.pkl` (~1 GB), `Results/ImageResults/` (~324 MB). Place them locally before full re-runs.

---

## Step-by-step: what was done (evaluation campaign)

### Step 1 — Environment & integrity
- Use conda env `VideoForgeryCPU` (Python 3.8, TensorFlow/Keras 2.10).
- Put `Library\bin` on `PATH` (required for scipy/skimage).
- **Integrity:** vendored `mealpy/metrics.py` was found to discard model predictions and inject random label flips. All Optimized scores use `Optimized/metrics_fixed.py` (sklearn only). See `Optimized/INTEGRITY_FINDING.md`.

### Step 2 — Multi-model baseline (epochs = 2)
```powershell
python -u Optimized\evaluate_multi.py --epochs 2 --train-pcts "0.8,0.9"
```
Models: DCNN, EfficientNetV2B0, MobileNetV2, STIDNet, P1-Proposed (3D-CNN+LSTM+SCAM+MUSE).  
Outputs: `Optimized/results/evaluation_multi_ep2.*`

### Step 3 — More epochs (20)
```powershell
python -u Optimized\evaluate_multi.py --epochs 20 --train-pcts "0.8,0.9"
```
Outputs: `evaluation_multi_ep20.*`

### Step 4 — Epochs = 50
```powershell
python -u Optimized\evaluate_multi.py --epochs 50 --train-pcts "0.8,0.9"
```
Outputs: `evaluation_multi_ep50.*`

### Step 5 — Epochs = 100
```powershell
python -u Optimized\evaluate_multi.py --epochs 100 --train-pcts "0.8,0.9"
```
Outputs: `evaluation_multi_ep100.*`  
Finding: more epochs alone did **not** reliably reach 95–99% honest test accuracy on N=50.

### Step 6 — Class weights + oversampling (epochs = 20)
```powershell
python -u Optimized\evaluate_multi.py --epochs 20 --class-weight --oversample --tag bal
```
Outputs: `evaluation_multi_ep20_bal_cw_os.*`  
Best balanced signal: MobileNetV2 BalAcc ≈ 81% @80% train.

### Step 7 — Minority weight-scale tuning
```powershell
python -u Optimized\run_weight_tuned.py
```
Sweeps `minority_scale` and `oversample_ratio` for deep + RF/GBM models.  
Log: `Optimized/logs/weight_tune.log`, summary: `results/weight_tune_best_summary.txt`.

### Step 8 — Full Word report
```powershell
python Optimized\build_results_docx.py
```
Produces: `Optimized/Paper1_MultiModel_Evaluation_Full_Report.docx` (all steps + tables + figures).

---

## Dataset note
- Cached features: `Features/Features.pkl` → proposed shape `(50, 10, 128, 128, 12)`, labels 29/21, majority baseline **58%**.
- Test sets at 80%/90% train are tiny (n=10 / n=5) → high variance.
- Paper reference `Analysis/TP/COM_A.npy` ~93% Acc is **reference only** (may use original metric path).

---

## How to run (quick start)

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd path\to\Video-Forensic-Paper-1-for-postdoc

# 1) Put Features.pkl into Features\ (not shipped)
# 2) Multi-model evaluation
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 20 --train-pcts "0.8,0.9"

# 3) With class balance
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 20 --class-weight --oversample --tag bal

# 4) Driver checks / plots / GUI (optional)
& "$E\python.exe" -u .claude\skills\run-video-forgery-paper1\driver.py check
```

---

## Postdoc context
- Offer: RIITC-Postdoc-2027-B03 (NKUST / RIITC)
- Topic: unified multi-modal deep learning for semantic media forensics
- Paper 1 = attention + LDZP/flow visual stream; Paper 2 = OM²AHL-BiG hybrid stream

## Authors (from postdoc offer letter)
- Dr. Abhishek Thakur — Chitkara University (primary); NKUST RIITC postdoc (secondary)
- Prof. Vishal Jain — VIPS-TC (co-supervisor)
- Prof. Chin-Shiuh Shieh — NKUST RIITC (supervisor)
