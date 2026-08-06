# Video Forensic Paper 1 for Postdoc

**SMA-CLMPNet** — Spatial Multiscale Attention enabled Convolutional Distributed Memory Network for intra-frame video forgery detection.

## Primary research paper (GENUINE results only)
`Optimized/Paper1_SMA_CLMPNet_Genuine_Research_Paper.docx`

All tables use sklearn `metrics_fixed.py`. Fabricated Analysis/COM_A and Results/TP–KF artefacts are **removed**.

## Fresh multi-model re-runs
- `Optimized/results/evaluation_multi_ep20_genuine.*`
- `Optimized/results/evaluation_multi_ep20_genuine_bal_cw_os.*`

## Authors (RIITC-Postdoc-2027-B03)
Dr. Abhishek Thakur · Prof. Vishal Jain · Prof. Chin-Shiuh Shieh

## Reproduce
```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 20 --train-pcts "0.8,0.9" --tag genuine
& "$E\python.exe" -u Optimized\evaluate_multi.py --epochs 20 --class-weight --oversample --tag genuine_bal
python Optimized\build_full_research_paper.py
```
