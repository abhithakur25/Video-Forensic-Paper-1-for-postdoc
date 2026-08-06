# Video Forensic Paper 1 for Postdoc

**Design and development of video forgery model using deep learning with attention mechanisms**

Private research package for Paper 1 (LDZP + optical flow + attention / 3D-CNN–LSTM–MUSE–SCAM), with a multi-model evaluation package under **`Optimized/`**.

> **GENUINE results only.** All reported scores use `Optimized/metrics_fixed.py` (sklearn).  
> Fabricated mealpy-based Analysis/COM_A and Results/TP–KF artefacts have been **removed**.

**Full genuine article:** [`Optimized/Paper1_Genuine_Results_Article.docx`](Optimized/Paper1_Genuine_Results_Article.docx)

---

## Genuine multi-model evaluation (re-run 2026-08-06)

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd <this-repo>
# Place Features.pkl into Features\ (not shipped — ~1 GB)

# Baseline (epochs=20)
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 20 --train-pcts "0.8,0.9" --tag genuine

# Class-weight + oversample
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 20 --class-weight --oversample --tag genuine_bal
```

Outputs (genuine):
- `Optimized/results/evaluation_multi_ep20_genuine.*`
- `Optimized/results/evaluation_multi_ep20_genuine_bal_cw_os.*`
- `Optimized/figures/`

### Models
DCNN · EfficientNetV2B0 · MobileNetV2 · STIDNet · P1-Proposed (3D-CNN+LSTM+SCAM+MUSE)

### Data
- proposed features `(50, 10, 128, 128, 12)`, labels 29/21, majority baseline **58%**
- Prefer **BalAcc / F1** over Acc on tiny test sets

### Integrity
Do **not** use `SubFunctions/Evaluate.py` → `mealpy.metrics` for reporting. See `Optimized/INTEGRITY_FINDING.md`.

---

## Structure

```
Optimized/          # multi-model eval + genuine results + article
SubFunctions/       # core pipeline
GUI.py Main.py
Features/           # put Features.pkl here locally
Results/            # non-fabricated residual assets only (TP/KF removed)
```

---

## Authors (RIITC-Postdoc-2027-B03)

- Dr. Abhishek Thakur — Chitkara University; NKUST RIITC postdoc  
- Prof. Vishal Jain — VIPS-TC (co-supervisor)  
- Prof. Chin-Shiuh Shieh — NKUST RIITC (supervisor)  

Related: [Video-Forensic-Paper-2-for-postdoc](https://github.com/abhithakur25/Video-Forensic-Paper-2-for-postdoc)
