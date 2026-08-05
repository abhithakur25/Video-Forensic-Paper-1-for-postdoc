# Integrity finding — Paper 1 evaluation metrics

## Issue
`mealpy/metrics.py` (vendored) overrides `_check_targets()` so that
`confusion_matrix(y_true, y_pred)` **discards model predictions** and replaces
them with ground-truth labels plus a random fraction of flips
(`per = random.uniform(0.090242, 0.45245235634)`).

Consequently every accuracy / sensitivity / specificity / precision / F1 value
produced through `SubFunctions/Evaluate.py → mealpy.metrics.confusion_matrix`
is **not** a true model score. This affects published `Analysis/*.npy` figures
when regenerated through the original path.

## Fix used in this package
`Optimized/metrics_fixed.py` scores with **sklearn** only
(`confusion_matrix`, `accuracy_score`, `balanced_accuracy_score`, etc.).

All numbers in `Optimized/results/` are machine-verified under this honest metric
path. Published `Analysis/TP/COM_*.npy` rows are kept only as **reference**
figures from the original paper artefacts, not as re-measured scores.
