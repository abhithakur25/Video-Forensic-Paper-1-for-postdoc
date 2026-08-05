# Comparison with other work on video forgery detection

Three things are compared here, and they must not be conflated:

1. **Measured in this repository** — the paper's own models and current
   architectures, run on the 50-video feature set, scored with
   `Optimized/metrics_fixed.py`. These are real measurements.
2. **Claimed by this paper** — the figures in the manuscript. These are
   outputs of the tampered metric and are not measurements of anything.
   See [`INTEGRITY_FINDING.md`](INTEGRITY_FINDING.md).
3. **Reported in the published literature** — figures from other groups on
   FaceForensics++. **These were not reproduced here.** They are quoted for
   context only, and the experimental setting differs so sharply from this
   project's that a direct numeric comparison would be meaningless. The
   explanation of why is the useful part of this document.

---

## The setting is not comparable, and that is the finding

FaceForensics++ (Rössler et al., ICCV 2019) ships 1,000 original YouTube
videos and 1,000 manipulated videos for each of four manipulation methods
(Deepfakes, Face2Face, FaceSwap, NeuralTextures), at three compression levels
(raw, c23/HQ, c40/LQ). Published detectors are trained at the **frame** level
on cropped faces, which turns ~2,000 videos into a training set on the order of
10⁵–10⁶ face crops.

This project evaluates on **50 videos** — 29 authentic, 21 forged — reduced to
one pre-extracted feature tensor per video. That is:

| | Published FF++ work | This project |
|---|---|---|
| Videos | ~2,000 per manipulation pairing | 50 total |
| Training unit | face crop per frame | one feature vector per video |
| Training samples | 10⁵–10⁶ | 19–44 |
| Test samples | 10⁴–10⁵ frames | 6–31 videos |
| Class balance | balanced by construction | 29 / 21 |

The training split at the 90% setting is 44 videos, and the test split is
**six**. One misclassification moves accuracy by 16.67 points. No experiment
of this size can resolve differences between detection methods, whatever the
methods are.

For orientation only, detectors of the XceptionNet family are commonly reported
in the literature to reach the low-to-mid 90s and above in frame-level accuracy
on FF++ c23, with performance degrading substantially at c40 compression and
degrading further under cross-manipulation transfer. **Treat these as context,
not as a baseline this project was measured against** — no such experiment was
run here, and this repository contains no trained checkpoint of any published
detector.

---

## What was actually measured here

Every number in this section was produced in this repository with a real
confusion matrix. Full tables: [`RESULTS.md`](RESULTS.md).

The headline: **no model separates the two classes.** Balanced accuracy near
50% with sensitivity 100 / specificity 0 (or the reverse) means the classifier
emits one label for every input. That describes most of the models the paper
compares, including its own proposed SMA-CLMPNet.

Three independent lines of evidence agree:

1. **The paper's models, correctly scored** — collapse to a constant
   prediction at 50.00% balanced accuracy.
2. **Current-generation backbones** (EfficientNetV2-S, ConvNeXt-Tiny,
   MobileNetV3-Large, ResNet-RS-50), as frozen ImageNet extractors with
   trained heads — 45–60% balanced accuracy, i.e. chance.
3. **Classical models under nested cross-validation** across richer
   representations (multi-scale spatial pooling, per-channel histograms,
   frame-to-frame temporal deltas) — see the search results in
   `optimize_v2.json`, permutation-tested.

The third is the decisive one, because classical models on 50 samples are not
data-starved the way a 3D CNN is, and a permutation test establishes what score
is achievable on **shuffled labels**. If the honest score cannot beat the 95th
percentile of that null, the features carry no signal that any model can use.

---

## Why the architecture is not the problem

It is tempting to read "our model gets chance accuracy" as an argument for a
better architecture. The evidence says otherwise:

- The **smallest** backbone tested (MobileNetV3-Large, 3.0 M frozen
  parameters) scored highest among the four. When capacity helps, bigger
  models win; when there is no signal, extra capacity only fits noise.
- Optimising the SMA-CLMPNet **training recipe** while holding the
  architecture fixed — batch 32→8, input standardisation, class weights,
  cosine-decayed learning rate over a longer budget — did not produce a
  discriminating classifier. It changed which class the network collapses onto.
- Classical models with explicit regularisation and feature selection, given
  representations that preserve spatial and temporal structure, did not beat
  the permutation null.

The limiting factor is the data: 50 videos summarised to one vector each.

---

## What would actually improve results

In descending order of expected effect:

1. **Use the real FaceForensics++ videos.** The `DATASET/` directory is empty;
   access is form-gated and the download script is emailed. Training at the
   frame level on the full c23 FaceSwap subset changes the sample count by
   three to four orders of magnitude. Nothing else on this list matters as
   much.
2. **Train at frame level, evaluate at video level.** Aggregate per-frame
   scores into a video decision. This is what published work does, and it is
   what turns 1,000 videos into a usable training set.
3. **Fine-tune a modern backbone on face crops**, rather than freezing an
   ImageNet extractor over whole-frame feature maps. Freezing was the correct
   choice *here* only because 19–44 samples cannot fine-tune anything.
4. **Report balanced accuracy and a confusion matrix, always.** On a 29/21
   corpus, accuracy alone hides a constant classifier at 58%.
5. **Report cross-manipulation and cross-compression results.** Within-dataset
   FF++ numbers are the easy case, and generalisation is where detectors
   actually fail.
6. **Fix the scoring before anything else.** Until `mealpy/metrics.py` is
   replaced, no experiment run through `SubFunctions/Evaluate.py` can be
   interpreted at all.
