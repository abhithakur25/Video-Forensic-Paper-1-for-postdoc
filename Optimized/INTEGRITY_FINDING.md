# Integrity finding: the reported metrics are fabricated

**Status: confirmed and reproducible. Every accuracy, sensitivity, specificity,
precision, F1 and ROC point produced by this pipeline as delivered is a random
number, independent of the models, the features and the training.**

Found 2026-08-05 while adding current-generation backbones to the comparison.

---

## What triggered the investigation

Four architecturally unrelated backbones — EfficientNetV2-S, ConvNeXt-Tiny,
MobileNetV3-Large, ResNet-RS-50 — returned **byte-identical** scores on all six
training splits and all five metrics:

```
EfficientNetV2S     96.8  92.3  95.2 100.0 100.0 100.0   mean 97.39
ConvNeXtTiny        96.8  92.3  95.2 100.0 100.0 100.0   mean 97.39
MobileNetV3Large    96.8  92.3  95.2 100.0 100.0 100.0   mean 97.39
ResNetRS50          96.8  92.3  95.2 100.0 100.0 100.0   mean 97.39
```

Their embeddings are demonstrably different (across-sample standard deviations
0.15 / 0.42 / 0.34 / 0.25), and their raw predictions disagree on 12–19 of 31
test samples. Identical scores from different predictions is impossible for a
real metric.

## Root cause

`SubFunctions/Evaluate.py` computes every metric from
`mealpy.metrics.confusion_matrix`. The vendored copy of that library has been
modified. `mealpy/metrics.py:16-75`, inside `_check_targets()`, immediately
before the confusion matrix is built:

```python
if perf:
    per = random.uniform(0.065242, 0.35245235634)
else:
    per = random.uniform(0.090242, 0.45245235634)

...

y = np.concatenate(dat)
y_true = shuffle(y, random_state=0)
y_pred = y_true.copy()                                   # <-- predictions discarded
va = random.sample(range(1, len(y_true)), int(len(y_true) * per))
for i in va:
    y_pred[i] = (random.sample(range(0, n), 1))[0]       # <-- corrupt a random fraction
```

The classifier's `y_pred` argument is overwritten on the third line with a copy
of the ground truth, then a random fraction `per` of its entries is randomised.
The confusion matrix is then computed between the labels and that synthetic
vector. Because a randomised binary label is still correct half the time,
expected accuracy is `1 - per/2`, i.e. uniform on roughly **0.77 to 0.955** —
precisely the band the paper reports.

The `perf=True` branch draws from a *narrower, more favourable* range
(0.065–0.352, expected accuracy 0.82–0.97).

Upstream `mealpy` contains no such code. The genuine function survives,
commented out, at `mealpy/metrics.py:285`.

## Demonstration

`Optimized/metrics_fixed.py` carries a self-test; the table below is from
`SubFunctions.Evaluate.Evaluation_Metrics` as delivered, `y_true` fixed at 15
zeros and 16 ones:

| Predictor | True accuracy | Reported, three consecutive calls |
|---|---|---|
| Perfect | 1.000 | 0.839, 0.935, 0.806 |
| Inverted (every prediction wrong) | 0.000 | 0.871, 0.710, 0.968 |
| Constant, all class 0 | 0.484 | 0.774, 0.774, 0.935 |
| Uniform random | 0.581 | 0.806, 0.935, 0.839 |

400 calls with a **perfect** predictor: min 0.645, max 1.000, mean 0.878.
A correct metric returns 1.000 every time. Two identical calls disagree, so the
function is not even deterministic.

## Blast radius

| Affected | How |
|---|---|
| §5.6.1 comparison vs training percentage | `Analysis.py:184` → `Evaluation_Metrics` |
| §5.6.2 k-fold comparison | `Analysis.py:233` → `Evaluation_Metrics` |
| §5.8 statistical analysis (best / mean / variance) | derived from the same arrays |
| All ROC curves | `Analysis.py:286-291` → `Evaluation_Metrics1`, same tampered matrix |
| `Analysis/` and `Analysis1/` `.npy` arrays shipped with the project | same |
| The reproduction run of 2026-08-04/05 (`Analysis1/TP`, `Analysis1/KF`) | same — the models trained genuinely, but scoring was fabricated |

The models themselves train normally. Only the scoring is fake, so re-scoring
real predictions with a correct metric recovers genuine results without
touching `SubFunctions/Model.py`.

### Beyond Paper 1

The same tampered file is present in the Paper 2 delivery. Searching for the
signature line across the working tree:

```
./CODE_28-04-2025_Paper1/mealpy/metrics.py     <- Paper 1 (and its copy here)
./CODE_05-08-2025_Paper2/mealpy/metrics.py     <- Paper 2
```

`CODE_05-08-2025_Paper2/SubFunctions/Evaluate.py:1` imports `confusion_matrix`
from it and calls it at lines 18 and 53 — the identical structure to Paper 1.
**Paper 2's reported metrics are fabricated by the same mechanism**, and PR-3
inherits from that work. Neither has been re-scored here; this document only
records that the same defect is present and on the same code path.

## Fix

`Optimized/metrics_fixed.py` reimplements `Evaluation_Metrics` and
`Evaluation_Metrics1`. The metric *formulas* in `SubFunctions/Evaluate.py` are
correct as written — including the convention that class 0 is the positive
class (`TP = cm[0,0]`) — so they are reproduced verbatim apart from
zero-division guards. Only the confusion matrix is replaced, with
`sklearn.metrics.confusion_matrix(..., labels=[0, 1])`.

Self-test result:

```
  perfect    expected 1.0000  got 1.0000  OK
  inverted   expected 0.0000  got 0.0000  OK
  all-zeros  expected 0.4839  got 0.4839  OK
  deterministic over 50 calls: OK
SELF-TEST PASSED
```

Nothing in `SubFunctions/` or `mealpy/` was edited — the tampered code is left
in place as evidence, and the corrected path is additive.

Balanced accuracy is reported alongside accuracy from here on, because the
corpus is 29 authentic / 21 forged and several models collapse onto the
majority class — a failure plain accuracy hides. EfficientNet at the 40% split
scores 58.06% accuracy with sensitivity 100% and specificity 0%: it predicts a
single class for every input, and balanced accuracy correctly reports 50.00%.

## What this means for the paper

The results sections do not describe measurements. They cannot be corrected by
re-wording; they need genuine numbers, which is what `Analysis1/TRUE` is being
regenerated to provide.

Whether the tampering was introduced upstream of this project or within it is
not something the code can answer, and this document makes no claim about it.
What is verifiable is that the file differs from upstream `mealpy`, that the
modification discards model predictions, and that every published number
depends on it.
