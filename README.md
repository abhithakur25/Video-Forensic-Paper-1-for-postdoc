# Video Forensic Paper 1 for Postdoc

**Design and development of video forgery model using deep learning with attention mechanisms**

Private research codebase for **Paper 1** (LDZP + optical flow + attention / 3D-CNN–LSTM–MUSE–SCAM), extended with a Paper-2-style multi-model optimization package under `Optimized/`.

> Full step-by-step evaluation log (all epoch runs, class weights, weight tuning, tables):  
> **[GITHUB_REPO_DESCRIPTION.md](GITHUB_REPO_DESCRIPTION.md)**  
> **[Optimized/Paper1_MultiModel_Evaluation_Full_Report.docx](Optimized/Paper1_MultiModel_Evaluation_Full_Report.docx)**

---

## Hardware / software

| Item | Recommendation |
|------|----------------|
| OS | Windows 10/11 |
| RAM | 16 GB+ |
| Python | 3.8 (conda env `VideoForgeryCPU`) |
| Deep learning | TensorFlow / Keras **2.10** |
| PATH | Must include conda `Library\bin` for scipy/skimage |

---

## Repository structure (high level)

```
├── GUI.py / Main.py          # Desktop app + plot regeneration
├── SubFunctions/             # Features, GradCAM, LDZP, Model, MUSE, SCAM
├── mealpy/                   # Vendored optimizers (metrics.py is NOT used for scoring)
├── Optimized/                # ★ Multi-model eval (honest sklearn metrics)
│   ├── evaluate_multi.py
│   ├── MultiModel.py
│   ├── metrics_fixed.py
│   ├── balance.py
│   ├── run_weight_tuned.py
│   ├── results/              # All CSV/TXT/NPY run logs
│   ├── figures/
│   └── Paper1_MultiModel_Evaluation_Full_Report.docx
├── Features/                 # Place Features.pkl here (not shipped — ~1 GB)
├── Analysis/ Analysis1/      # NPY paper artefacts (reference)
├── Results/                  # Plots/CSVs (ImageResults/ not shipped)
└── .claude/skills/.../driver.py
```

---

## Step-by-step: multi-model evaluation (what we ran)

### 1) Setup environment
```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd <this-repo>
# Copy Features.pkl into Features\  (required for Optimized eval)
```

### 2) Baseline multi-model (epochs = 2)
```powershell
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 2 --train-pcts "0.8,0.9"
```
→ `Optimized/results/evaluation_multi_ep2.txt|csv`

### 3) Longer training (epochs = 20 / 50 / 100)
```powershell
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 20 --train-pcts "0.8,0.9"
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 50 --train-pcts "0.8,0.9"
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 100 --train-pcts "0.8,0.9"
```

### 4) Class weights + minority oversampling
```powershell
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 20 --class-weight --oversample --tag bal
```
→ `evaluation_multi_ep20_bal_cw_os.*`

### 5) Minority weight-scale tuning
```powershell
& "$E\python.exe" -u Optimized\run_weight_tuned.py
```

### 6) Build the Word report of all steps
```powershell
python Optimized\build_results_docx.py
```

### 7) Optional GUI / plots driver
```powershell
& "$E\python.exe" -u .claude\skills\run-video-forgery-paper1\driver.py check
& "$E\python.exe" -u .claude\skills\run-video-forgery-paper1\driver.py plots
& "$E\python.exe" -u .claude\skills\run-video-forgery-paper1\driver.py gui
```

---

## Models compared (Optimized)

| Model | Role |
|-------|------|
| DCNN | Compact CNN baseline |
| EfficientNetV2B0 | Latest TF-2.10 backbone |
| MobileNetV2 | Mobile CNN baseline |
| STIDNet | Teacher–student (Paper 1 comparative) |
| P1-Proposed | 3D-CNN + dual LSTM + SCAM + MUSE |
| RF / GBM | Classical ensembles (weight-tune phase) |

**Metrics:** Acc, Sen, Spec, Prec, F1, **Balanced Acc** via `Optimized/metrics_fixed.py` only.

**Data:** 50 samples (29/21), majority baseline **58%**. Prefer BalAcc/F1 over Acc on tiny test sets.

---

## Integrity note

Original `mealpy/metrics.confusion_matrix` **must not** be used for reporting (randomized labels). See `Optimized/INTEGRITY_FINDING.md`.

---

## Postdoc / authors (offer letter RIITC-Postdoc-2027-B03)

- **Dr. Abhishek Thakur** — Chitkara University (primary); NKUST RIITC postdoc (secondary)  
- **Prof. Vishal Jain** — VIPS-TC (co-supervisor)  
- **Prof. Chin-Shiuh Shieh** — NKUST RIITC (supervisor)  

Related repo: [Video-Forensic-Paper-2-for-postdoc](https://github.com/abhithakur25/Video-Forensic-Paper-2-for-postdoc)

---

## License / use

Research / academic use for the postdoctoral programme. Contact the corresponding author for dataset access and collaboration.
