# Results — Paper 1, corrected scoring

Generated 2026-08-05 11:22:55.

> **The scores this project produces as delivered are fabricated.** The vendored `mealpy/metrics.py` discards model predictions and invents them. See [`INTEGRITY_FINDING.md`](INTEGRITY_FINDING.md). Everything below headed *measured* was scored with `Optimized/metrics_fixed.py`.

Splits completed in this run: 40%. Baseline budget 10 epochs; SMA-CLMPNet-Opt 30 epochs at batch 8.


## The gap: reported by the pipeline vs actually measured

Accuracy, mean over the splits evaluated in both runs. The left column is what the delivered code prints; it is not a measurement of anything.

| Model        | Fabricated | Measured acc. | Measured bal. acc. | Δ (pp) |
|--------------|------------|---------------|--------------------|--------|
| EfficientNet | 87.10      | 41.94         | 50.00              | -45.16 |
| STIDNet      | 100.00     | 58.06         | 51.07              | -41.94 |
| DCNN         | 87.10      | 58.06         | 50.00              | -29.03 |
| GLCM         | 90.32      | 54.84         | 55.77              | -35.48 |
| MUSE-CLMPNet | 96.77      | 58.06         | 50.00              | -38.71 |
| SCAM-CLMPNet | 93.55      | 48.39         | 54.49              | -45.16 |
| SMA-CLMPNet  | 87.10      | 58.06         | 50.00              | -29.03 |


## Measured accuracy by training percentage

| Model            | 40%   | 50% | 60% | 70% | 80% | 90% | Mean acc. | Mean bal. acc. |
|------------------|-------|-----|-----|-----|-----|-----|-----------|----------------|
| EfficientNet     | 41.94 | —   | —   | —   | —   | —   | 41.94     | 50.00          |
| STIDNet          | 58.06 | —   | —   | —   | —   | —   | 58.06     | 51.07          |
| DCNN             | 58.06 | —   | —   | —   | —   | —   | 58.06     | 50.00          |
| GLCM             | 54.84 | —   | —   | —   | —   | —   | 54.84     | 55.77          |
| MUSE-CLMPNet     | 58.06 | —   | —   | —   | —   | —   | 58.06     | 50.00          |
| SCAM-CLMPNet     | 48.39 | —   | —   | —   | —   | —   | 48.39     | 54.49          |
| SMA-CLMPNet      | 58.06 | —   | —   | —   | —   | —   | 58.06     | 50.00          |
| EfficientNetV2S  | 51.61 | —   | —   | —   | —   | —   | 51.61     | 48.72          |
| ConvNeXtTiny     | 61.29 | —   | —   | —   | —   | —   | 61.29     | 55.98          |
| MobileNetV3Large | 61.29 | —   | —   | —   | —   | —   | 61.29     | 58.12          |
| ResNetRS50       | 38.71 | —   | —   | —   | —   | —   | 38.71     | 35.47          |
| SMA-CLMPNet-Opt  | 45.16 | —   | —   | —   | —   | —   | 45.16     | 52.78          |

Chance is 50.00%. Majority-class-always is 58.00% accuracy and 50.00% balanced accuracy on this 29/21 corpus — any model at 50.00% balanced accuracy has learned nothing and is predicting one class for every input.


## Measured means, every metric

| Model            | Accuracy | Sensitivity | Specificity | Precision | F1-score | Balanced acc. |
|------------------|----------|-------------|-------------|-----------|----------|---------------|
| EfficientNet     | 41.94    | 0.00        | 100.00      | —         | 0.00     | 50.00         |
| STIDNet          | 58.06    | 94.44       | 7.69        | 58.62     | 72.34    | 51.07         |
| DCNN             | 58.06    | 100.00      | 0.00        | 58.06     | 73.47    | 50.00         |
| GLCM             | 54.84    | 50.00       | 61.54       | 64.29     | 56.25    | 55.77         |
| MUSE-CLMPNet     | 58.06    | 100.00      | 0.00        | 58.06     | 73.47    | 50.00         |
| SCAM-CLMPNet     | 48.39    | 16.67       | 92.31       | 75.00     | 27.27    | 54.49         |
| SMA-CLMPNet      | 58.06    | 100.00      | 0.00        | 58.06     | 73.47    | 50.00         |
| EfficientNetV2S  | 51.61    | 66.67       | 30.77       | 57.14     | 61.54    | 48.72         |
| ConvNeXtTiny     | 61.29    | 88.89       | 23.08       | 61.54     | 72.73    | 55.98         |
| MobileNetV3Large | 61.29    | 77.78       | 38.46       | 63.64     | 70.00    | 58.12         |
| ResNetRS50       | 38.71    | 55.56       | 15.38       | 47.62     | 51.28    | 35.47         |
| SMA-CLMPNet-Opt  | 45.16    | 5.56        | 100.00      | 100.00    | 10.53    | 52.78         |


## SMA-CLMPNet: published training recipe vs optimised

Identical architecture — the authors' MUSE block, SCAM attention, 3D convolution stack and dual LSTM, rebuilt layer for layer. Only the training recipe differs: batch 32→8 (44 training samples give 2 gradient steps per epoch at batch 32), inputs standardised with training-split statistics only, class weights for the 29/21 imbalance, and a cosine-decayed learning rate over a longer budget.

| Training % | Base acc. | Opt acc. | Δ      | Base bal. | Opt bal. | Δ     |
|------------|-----------|----------|--------|-----------|----------|-------|
| 40%        | 58.06     | 45.16    | -12.90 | 50.00     | 52.78    | +2.78 |
| **Mean**   | 58.06     | 45.16    | -12.90 | 50.00     | 52.78    | +2.78 |


## Reading these numbers honestly

The corpus is 50 videos, 29 authentic / 21 forged, and the test partition ranges from 31 videos at the 40% split down to 6 at the 90% split. One misclassification moves accuracy by 3.23 pp at 40% and 16.67 pp at 90%. Differences smaller than that are noise, and nothing here supports a significance claim. Reporting a 90%-split number from six test videos is not meaningful regardless of how it is scored.

BA-TFD is absent throughout: its ViTDCNN definition applies `MaxPooling2D(1, 1)`, which does not downsample, so the flattened 1,048,576-element vector entering `Dense(2048)` requires an 8.6 GB weight matrix and exhausts memory at every batch size tested.
