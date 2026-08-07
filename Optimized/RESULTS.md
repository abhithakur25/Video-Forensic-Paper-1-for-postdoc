# Results — Paper 1, corrected scoring

Generated 2026-08-07 10:17:37.

> **The scores this project produces as delivered are fabricated.** The vendored `mealpy/metrics.py` discards model predictions and invents them. See [`INTEGRITY_FINDING.md`](INTEGRITY_FINDING.md). Everything below headed *measured* was scored with `Optimized/metrics_fixed.py`.

Splits completed in this run: 40%, 50%, 60%, 70%, 80%, 90%. Baseline budget 10 epochs; SMA-CLMPNet-Opt 30 epochs at batch 8.



## Measured accuracy by training percentage

| Model            | 40%   | 50%   | 60%   | 70%   | 80%   | 90%   | Mean acc. | Mean bal. acc. |
|------------------|-------|-------|-------|-------|-------|-------|-----------|----------------|
| EfficientNet     | 41.94 | 57.69 | 57.14 | 56.25 | 54.55 | 50.00 | 52.93     | 50.00          |
| STIDNet          | 58.06 | 57.69 | 57.14 | 75.00 | 54.55 | 66.67 | 61.52     | 56.53          |
| DCNN             | 58.06 | 57.69 | 42.86 | 56.25 | 54.55 | 33.33 | 50.46     | 47.22          |
| GLCM             | 54.84 | 53.85 | 42.86 | 43.75 | 45.45 | 50.00 | 48.46     | 48.53          |
| MUSE-CLMPNet     | 58.06 | 57.69 | 57.14 | 56.25 | 54.55 | 50.00 | 55.62     | 50.00          |
| SCAM-CLMPNet     | 48.39 | 46.15 | 57.14 | 43.75 | 54.55 | 33.33 | 47.22     | 47.89          |
| SMA-CLMPNet      | 58.06 | 57.69 | 57.14 | 43.75 | 54.55 | 50.00 | 53.53     | 50.00          |
| EfficientNetV2S  | 51.61 | 46.15 | 47.62 | 56.25 | 72.73 | 66.67 | 56.84     | 56.10          |
| ConvNeXtTiny     | 61.29 | 61.54 | 42.86 | 50.00 | 45.45 | 50.00 | 51.86     | 50.41          |
| MobileNetV3Large | 61.29 | 61.54 | 61.90 | 62.50 | 54.55 | 66.67 | 61.41     | 59.75          |
| ResNetRS50       | 38.71 | 42.31 | 47.62 | 56.25 | 45.45 | 50.00 | 46.72     | 45.23          |
| SMA-CLMPNet-Opt  | 45.16 | 57.69 | 57.14 | 62.50 | 54.55 | 50.00 | 54.51     | 51.92          |

Chance is 50.00%. Majority-class-always is 58.00% accuracy and 50.00% balanced accuracy on this 29/21 corpus — any model at 50.00% balanced accuracy has learned nothing and is predicting one class for every input.


## Measured means, every metric

| Model            | Accuracy | Sensitivity | Specificity | Precision | F1-score | Balanced acc. |
|------------------|----------|-------------|-------------|-----------|----------|---------------|
| EfficientNet     | 52.93    | 83.33       | 16.67       | 55.13     | 59.19    | 50.00         |
| STIDNet          | 61.52    | 93.52       | 19.54       | 60.65     | 72.89    | 56.53         |
| DCNN             | 50.46    | 72.22       | 22.22       | 51.98     | 53.76    | 47.22         |
| GLCM             | 48.46    | 57.41       | 39.65       | 54.62     | 49.38    | 48.53         |
| MUSE-CLMPNet     | 55.62    | 100.00      | 0.00        | 55.62     | 71.44    | 50.00         |
| SCAM-CLMPNet     | 47.22    | 51.11       | 44.68       | 57.72     | 44.35    | 47.89         |
| SMA-CLMPNet      | 53.53    | 83.33       | 16.67       | 55.49     | 59.44    | 50.00         |
| EfficientNetV2S  | 56.84    | 55.28       | 56.92       | 67.59     | 58.26    | 56.10         |
| ConvNeXtTiny     | 51.86    | 57.41       | 43.41       | 55.63     | 55.82    | 50.41         |
| MobileNetV3Large | 61.41    | 78.70       | 40.79       | 62.69     | 69.17    | 59.75         |
| ResNetRS50       | 46.72    | 52.87       | 37.59       | 51.91     | 52.01    | 45.23         |
| SMA-CLMPNet-Opt  | 54.51    | 82.41       | 21.43       | 63.49     | 61.07    | 51.92         |


## Measured k-fold comparison

Stratified k-fold, k = 6, 7, 8, 1 fold per k, scored with `metrics_fixed.py`. The published `KFAnalysis` could not be used: `Analysis.py:355` indexes `data['image']`, a key `ReadDataset` never stores.

| Model            | k=6   | k=7   | k=8   | Mean acc. | Mean bal. acc. |
|------------------|-------|-------|-------|-----------|----------------|
| STIDNet          | 66.67 | 75.00 | 71.43 | 71.03     | 70.56          |
| GLCM             | 66.67 | 87.50 | 57.14 | 70.44     | 65.28          |
| EfficientNetV2S  | 55.56 | 62.50 | 71.43 | 63.16     | 58.61          |
| ConvNeXtTiny     | 44.44 | 62.50 | 57.14 | 54.70     | 50.28          |
| DCNN             | 55.56 | 62.50 | 57.14 | 58.40     | 50.00          |
| EfficientNet     | 55.56 | 62.50 | 57.14 | 58.40     | 50.00          |
| MUSE-CLMPNet     | 55.56 | 62.50 | 57.14 | 58.40     | 50.00          |
| SMA-CLMPNet-Opt  | 44.44 | 62.50 | 57.14 | 54.70     | 50.00          |
| SMA-CLMPNet      | 55.56 | 62.50 | 57.14 | 58.40     | 50.00          |
| SCAM-CLMPNet     | 33.33 | 50.00 | 28.57 | 37.30     | 43.61          |
| MobileNetV3Large | 33.33 | 12.50 | 28.57 | 24.80     | 23.33          |
| ResNetRS50       | 11.11 | 12.50 | 14.29 | 12.63     | 10.83          |

Each test fold holds 5-9 of the 50 videos, so one misclassification moves accuracy by 11-20 pp. No difference in this table is resolvable at that granularity.



## SMA-CLMPNet: published training recipe vs optimised

Identical architecture — the authors' MUSE block, SCAM attention, 3D convolution stack and dual LSTM, rebuilt layer for layer. Only the training recipe differs: batch 32→8 (44 training samples give 2 gradient steps per epoch at batch 32), inputs standardised with training-split statistics only, class weights for the 29/21 imbalance, and a cosine-decayed learning rate over a longer budget.

| Training % | Base acc. | Opt acc. | Δ      | Base bal. | Opt bal. | Δ     |
|------------|-----------|----------|--------|-----------|----------|-------|
| 40%        | 58.06     | 45.16    | -12.90 | 50.00     | 52.78    | +2.78 |
| 50%        | 57.69     | 57.69    | +0.00  | 50.00     | 50.00    | +0.00 |
| 60%        | 57.14     | 57.14    | +0.00  | 50.00     | 50.00    | +0.00 |
| 70%        | 43.75     | 62.50    | +18.75 | 50.00     | 58.73    | +8.73 |
| 80%        | 54.55     | 54.55    | +0.00  | 50.00     | 50.00    | +0.00 |
| 90%        | 50.00     | 50.00    | +0.00  | 50.00     | 50.00    | +0.00 |
| **Mean**   | 53.53     | 54.51    | +0.97  | 50.00     | 51.92    | +1.92 |


## Model selection over richer representations

Protocol: nested CV, outer 5-fold, inner 4-fold, balanced accuracy. The deep models above are trained on a single deterministic split; this section instead does nested cross-validation, so the reported score is estimated on folds the hyper-parameter selection never saw. Representations preserve what a channel mean destroys: multi-scale spatial layout, per-channel distributions, and frame-to-frame temporal change.

| Representation                 | Model       | Nested bal. acc. | SD across folds |
|--------------------------------|-------------|------------------|-----------------|
| proposed: temporal delta stats | logreg-l1   | 77.17            | ±15.43          |
| proposed: mean-frame grid pool | extra-trees | 65.33            | ±8.75           |
| proposed: spatial + temporal   | svm-rbf     | 64.17            | ±7.73           |
| proposed: temporal delta stats | extra-trees | 62.50            | ±12.64          |
| ALL (grid+hist+temporal+GLCM)  | rf          | 62.50            | ±25.95          |
| EfficientNetV2S embedding      | gaussian-nb | 62.33            | ±6.78           |
| proposed: spatial + temporal   | logreg-l1   | 61.67            | ±12.19          |
| proposed: temporal delta stats | knn         | 61.50            | ±8.73           |
| proposed: mean-frame grid pool | rf          | 61.17            | ±15.01          |
| proposed: spatial + temporal   | svm-lin     | 60.67            | ±7.04           |
| proposed: temporal delta stats | logreg-l2   | 60.17            | ±15.93          |
| proposed: spatial + temporal   | logreg-l2   | 59.00            | ±12.60          |

**Permutation test on the winner** (logreg-l1 on proposed: temporal delta stats): observed 77.17%, null mean 50.23%, null 95th percentile **63.51%**, p = **0.010**.

Verdict: signal above chance.


## Independent probe

A separate, simpler probe (repeated stratified 5-fold, logistic regression / RBF SVM / random forest on summary statistics) reached 59.90% balanced accuracy on 'comparative1 (chan mean+std)', against a null 95th percentile of 64.67% (p = 0.174).



## Reading these numbers honestly

The corpus is 50 videos, 29 authentic / 21 forged, and the test partition ranges from 31 videos at the 40% split down to 6 at the 90% split. One misclassification moves accuracy by 3.23 pp at 40% and 16.67 pp at 90%. Differences smaller than that are noise, and nothing here supports a significance claim. Reporting a 90%-split number from six test videos is not meaningful regardless of how it is scored.

BA-TFD is absent throughout: its ViTDCNN definition applies `MaxPooling2D(1, 1)`, which does not downsample, so the flattened 1,048,576-element vector entering `Dense(2048)` requires an 8.6 GB weight matrix and exhausts memory at every batch size tested.
