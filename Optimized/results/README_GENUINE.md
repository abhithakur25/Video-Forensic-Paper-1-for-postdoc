# Genuine results only

All CSVs/TXTs in this folder were produced by `Optimized/evaluate_multi.py` using `metrics_fixed.py` (sklearn).

## Fresh re-run (2026-08-06)
- `evaluation_multi_ep20_genuine.*` — epochs=20, no balance
- `evaluation_multi_ep20_genuine_bal_cw_os.*` — epochs=20, class_weight + oversample

## Archive genuine ladder (same metric path, earlier)
- ep2, ep20, ep50, ep100, ep20_bal_cw_os

## Removed (fabricated)
- Analysis/, Analysis1/, Results/TP, Results/KF, Results/RocAnalysis (mealpy metric artefacts)
