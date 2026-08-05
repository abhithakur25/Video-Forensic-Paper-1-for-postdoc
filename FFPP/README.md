# FaceForensics++ pipeline

Frame-level training, video-level evaluation, correct scoring. This is the
route to genuine 90%-range accuracy — not by tuning, but by training on three
to four orders of magnitude more data than the 50-video feature set.

| | Existing evaluation | This pipeline |
|---|---|---|
| Videos | 50 | ~2,000 (1,000 real + 1,000 manipulated) |
| Training unit | one feature vector per video | one face crop per sampled frame |
| Training samples | 19–44 | ~50,000–64,000 |
| Test samples | 6–31 videos | ~200–400 videos, thousands of crops |

**Status: built and smoke-tested; waiting on the dataset.** `DATASET/` is
empty and FF++ access is form-gated, so the download has to be done by you.

---

## 1. Get the data

Request access at **https://github.com/ondyari/FaceForensics** — you accept
their terms and they email back a `faceforensics_download_v4.py` script. It is
tied to your acceptance, so I can't do this step.

```bash
# c23 (HQ) is the standard benchmark compression
python faceforensics_download_v4.py <output_dir> -d original       -c c23 -t videos
python faceforensics_download_v4.py <output_dir> -d FaceSwap       -c c23 -t videos
# optionally the other three manipulations
python faceforensics_download_v4.py <output_dir> -d Deepfakes      -c c23 -t videos
python faceforensics_download_v4.py <output_dir> -d Face2Face      -c c23 -t videos
python faceforensics_download_v4.py <output_dir> -d NeuralTextures -c c23 -t videos
```

Expected layout (what the ingestion step looks for):

```
<root>/original_sequences/youtube/c23/videos/*.mp4            label 0
<root>/manipulated_sequences/FaceSwap/c23/videos/*.mp4        label 1
```

Roughly 40 GB for original + FaceSwap at c23.

## 2. Ingest — videos to cached face crops

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1

& "$E\python.exe" -u FFPP/ffpp_data.py `
    --root <root> --out FFPP/cache `
    --manipulations FaceSwap --compression c23 `
    --max-frames 32 --size 224
```

Samples 32 frames per video, detects the largest face, crops with a 30%
margin, resizes to 224², and caches to `FFPP/cache/`.

Size: 2,000 videos × 32 crops × 224² × 3 bytes ≈ **9.6 GB**. Drop
`--max-frames` to 16 to halve it. Use `--limit 40` for a quick trial first.

## 3. Train and evaluate

```powershell
# quick baseline: frozen backbone, head only
& "$E\python.exe" -u FFPP/ffpp_train.py --cache FFPP/cache `
    --backbone EfficientNetV2B0 --mode single --freeze --epochs 4

# the real run: fine-tune the last 40 layers, training-percentage table
& "$E\python.exe" -u FFPP/ffpp_train.py --cache FFPP/cache `
    --backbone EfficientNetV2B0 --mode tp --epochs 6 --trainable-from -40

# k-fold table
& "$E\python.exe" -u FFPP/ffpp_train.py --cache FFPP/cache `
    --backbone EfficientNetV2B0 --mode kfold --folds 5 --epochs 6
```

Backbones: `EfficientNetV2B0`, `EfficientNetV2S`, `MobileNetV3Large`,
`ConvNeXtTiny`, `Xception` (the FF++ paper's own choice).

Results land in `FFPP/results/<backbone>_<mode>.json`, with columns
`ACC, SEN, SPE, PRE, F1, BAL` at both frame and video level — the same
columns as the paper's tables.

**CPU vs GPU.** This machine is CPU-only. Fine-tuning EfficientNetV2B0 on
~50k crops will take days here; frozen-backbone mode is hours. On a single
modern GPU the full fine-tune is roughly an hour. If no GPU is available, run
`--freeze` first for a real baseline, then fine-tune where a GPU exists.

## 4. What protects the numbers

**Splits are grouped by source identity.** FF++ names manipulated clips
`<target>_<source>.mp4`, sharing footage with `<target>.mp4`. If frames from
one video — or the original and its manipulation — straddle the split, the
model learns to recognise the footage rather than the manipulation, and
accuracy goes near 100% while meaning nothing. This is the single most common
cause of implausible deepfake detection results.

Every split runs through `assert_disjoint()`, which **halts the run** on any
shared identity and prints the counts otherwise:

```
split integrity OK (tp 90%): 10 train / 2 test identities, 0 shared
```

**Video-level aggregation.** Frame probabilities are averaged per video and
thresholded once, so the reported figure is per video, comparable with the
paper's tables. Frame-level numbers are printed alongside.

**Scoring** goes through `Optimized/metrics_fixed.py`, never
`SubFunctions/Evaluate.py` — see `Optimized/INTEGRITY_FINDING.md`.

## 5. Verify before the data arrives

```powershell
& "$E\python.exe" -u FFPP/smoke_test.py
```

Builds a synthetic FF++ tree with the real directory and filename convention,
runs ingestion and all three training modes, and checks that the leakage guard
fires on shared identities and passes on disjoint ones. Last run:

```
1. building synthetic FF++ tree            24 videos
2. ingestion                               192 crops from 24 videos
3. leakage guard                           fires on shared, passes on disjoint
4. training (single split)                 frame 85.42  ->  VIDEO 100.00
5. training-percentage mode                6 splits
6. k-fold mode                             3 result files
SMOKE TEST PASSED - pipeline is ready for real data
```

Synthetic "forged" clips carry a periodic inter-frame discontinuity the
authentic ones lack, so above-chance scores confirm the plumbing works. It
tests the pipeline, not detection quality.

## 6. Expected outcome, stated in advance

Published FF++ c23 detectors of the Xception family report low-to-mid 90s
frame-level accuracy within-dataset. If this pipeline lands in that region,
it is consistent with the literature. If it lands near 99–100%, **suspect
leakage before celebrating** — check the integrity line, confirm the identity
grouping, and re-read `meta.json`.

Performance degrades on c40 compression and degrades further across
manipulation types. Reporting cross-manipulation and cross-compression results
is what distinguishes a credible detection paper from a within-dataset one.

## Files

| File | Purpose |
|---|---|
| `ffpp_data.py` | scan FF++ tree → sample frames → crop faces → cache |
| `ffpp_train.py` | fine-tune backbone, video-level eval, TP/k-fold tables |
| `smoke_test.py` | end-to-end verification on synthetic videos |
