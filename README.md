# Video Forensic — Paper 1 (Postdoc)

> ## ⚠️ The metrics this code produces are fabricated
>
> The vendored `mealpy/metrics.py` has been modified so that
> `_check_targets()` **discards the model's predictions** and replaces them
> with the ground truth plus a random fraction of flipped labels
> (`mealpy/metrics.py:70-75`). Every accuracy, sensitivity, specificity,
> precision, F1 and ROC point produced by the pipeline as delivered is a draw
> from a random number generator, independent of the models and the data.
> A perfect predictor scores 0.645–1.000 across repeated calls; an inverted
> predictor scores just as well.
>
> This affects the paper's results sections, the `.npy` arrays under
> `Analysis/` and `Analysis1/`, **and the reproduction figures previously
> recorded in this repository**, which were produced in good faith through the
> same function.
>
> Evidence, blast radius and fix: **[`Optimized/INTEGRITY_FINDING.md`](Optimized/INTEGRITY_FINDING.md)**.
> Corrected scores: **[`Optimized/RESULTS.md`](Optimized/RESULTS.md)**.
> Do not cite any number from this project that was not scored with
> `Optimized/metrics_fixed.py`.

**Design and Development of a Video Forgery Model Using Deep Learning with
Attention Mechanisms** — reference implementation, execution harness and
reproduction record.

Proposed model: **SMA-CLMPNet** — Spatial Multiscale Attention coupled
Convolutional distributed LSTM with Modified Pooling. Dataset:
**FaceForensics++** (FaceSwap subset, c23 compression).

This repository contains the original research code **unmodified**, plus an
automation harness (`driver.py`) that makes it runnable without a human at the
keyboard, and a full record of the evaluation that was run.

The authors' original README is preserved verbatim as
[`README_ORIGINAL.md`](README_ORIGINAL.md).

---

## 1. Why a harness is needed

Neither entry point can be driven from a shell:

| Entry point | Why it blocks |
|---|---|
| `Main.py` | Line 5 opens a modal `PySimpleGUI.popup_yes_no()`. Then `PlotResults()` defaults to `show=True`, producing ~40 blocking `plt.show()` windows. |
| `GUI.py` | Its only entry is `app.mainloop()`. "Select Video" opens a modal native file dialog (`filedialog.askopenfilename`). |

`.claude/skills/run-video-forgery-paper1/driver.py` bypasses both, drives the
real code, and captures results and screenshots.

---

## 2. Environment

Windows 11, **Python 3.8**, CPU only. The default `python` on most machines will
not work — TensorFlow 2.10 and numpy 1.21.6 have no builds for Python 3.12+.

```powershell
conda create -n VideoForgeryCPU python=3.8 -y
conda activate VideoForgeryCPU
conda install -y numpy=1.21.6 scipy=1.7.3 pandas=1.3.4 matplotlib=3.5.3 `
    seaborn=0.12.2 scikit-learn=1.0.2 scikit-image=0.19.3 -c conda-forge
pip install tensorflow==2.10.0 keras==2.10.0 opencv-python==4.8.0.76 `
    Pillow==9.5.0 termcolor tqdm customtkinter==5.1.3 "PySimpleGUI==4.60.5.1"
```

Three pins deviate from `requirements.txt` and are **not optional**:

- **`PySimpleGUI==4.60.5.1`** — `Main.py` imports PySimpleGUI but it is missing
  from `requirements.txt` entirely, and version `4.60.5` was withdrawn from PyPI
  when the project relicensed. `4.60.5.1` is the last free 4.x release.
- **`customtkinter==5.1.3`** — an unpinned install gives 6.x, whose API differs.
- **`scikit-image==0.19.3`** — `GetFeatures.py:22` imports `greycomatrix` /
  `greycoprops`, renamed to `gray*` in 0.20 and later removed.

### Running commands

Every command in this README assumes these two lines first:

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
```

**`$E\Library\bin` is mandatory.** Without it `import skimage` does not raise —
it *hard-crashes the interpreter* with exit code `-1066598273` (0xC0000409,
"stack buffer overrun") and no traceback. `-X faulthandler` traces it to
`skimage/color/colorconv.py:396` → `scipy.linalg.inv` → Windows exception
`0xc06d007f`, a DLL delay-load failure: conda keeps its MKL in `Library\bin`, and
launching `python.exe` by absolute path leaves that directory off `PATH`.
`conda activate VideoForgeryCPU` sets it too.

Always pass `python -u`. On a hard crash, buffered stdout is discarded and the
failure looks like silence.

---

## 3. Every executable file, and what it does

### 3.1 Entry points

| File | Role |
|---|---|
| **`Main.py`** | Research entry point. Asks "Complete Execution?" — **Yes** re-extracts features from `DATASET/` then runs the full analysis (~48 h); **No** only re-plots from stored `Analysis/*.npy` (~30 s). |
| **`GUI.py`** | customtkinter desktop app demonstrating the feature-extraction pipeline one stage at a time on a single video. |
| **`.claude/skills/run-video-forgery-paper1/driver.py`** | The automation harness. All reproduction below goes through it. |

### 3.2 `SubFunctions/` — the pipeline

Import order matters: `SubFunctions/__init__.py` re-exports `ReadDataset`,
`TPAnalysis`, `KFAnalysis` and `PlotResults`, so importing *anything* from the
package executes the whole chain.

| Module | Lines | What it does |
|---|---|---|
| **`GetData.py`** | 139 | `ReadDataset`. `exec=True` globs `DATASET/manipulated_sequences/FaceSwap/c23/videos/*.mp4` (label 1) and `DATASET/original_sequences/youtube/c23/videos/*.mp4` (label 0), extracts features and writes `Features/Features.pkl`. `exec=False` loads that pickle — this is the path used for evaluation. |
| **`GetPreprocessing.py`** | 47 | `Preprocessing`. Gradient-based key-frame selection, then Haar-cascade face ROI cropping (`Temp/haarcascade_frontalface_alt2.xml`). |
| **`GetFeatures.py`** | 268 | `FeatureExtraction`. Builds the feature families: Grad-CAM deep flow map, ResNet-101 statistical features (mean/variance/std/skew/kurtosis), VGG-16 + LDZP, and Lucas-Kanade optical flow. **Instantiates `ResNet101()` and `VGG16()` at module scope** — see §7. |
| **`GradCAM.py`** | 67 | `CAM` / `GradCAM` — gradient-weighted class activation maps over MobileNetV2. |
| **`LDZP.py`** | 197 | `LocalDirectionalZigZagPattern` — the LDZP texture descriptor. |
| **`MUSE.py`** | 79 | Multi-excitation attention block (the MUSE-CLMPNet ablation). |
| **`SCAM.py`** | 126 | Spatial, channel and joint spatial-channel attention layers (the SCAM-CLMPNet ablation). |
| **`Model.py`** | 526 | `Network` — all eight classifiers: `EfficientNet`, `STIDNet`, `CNN` (DCNN), `GLCM`, `ViTDCNN` (BA-TFD) and `ThreeDCNNLSTM(opt=1/2/3)` = MUSE- / SCAM- / **SMA-CLMPNet**. Also `Distiller` for knowledge distillation. |
| **`Analysis.py`** | 500 | `TPAnalysis` (training percentage 40–90 %), `KFAnalysis` (k = 6…10), plus `RocAnalysis`. |
| **`Evaluate.py`** | 77 | `Evaluation_Metrics` — confusion matrix → accuracy, sensitivity, specificity, precision, F1. ⚠️ **Its output is fabricated**: it scores through the tampered `mealpy.metrics.confusion_matrix`. The formulas are correct; the matrix is not. Use `Optimized/metrics_fixed.py`. |
| **`VisualizeResults.py`** | 502 | `PlotResults`. `show=True` blocks on `plt.show()`; `show=False, save=True` writes PNG + CSV under `Results/`. |
| **`mealpy/`** | — | Vendored metaheuristic optimiser package (**local copy, not the PyPI one**). Because it is vendored, the working directory must be the project root. ⚠️ `metrics.py` is modified relative to upstream; `_check_targets()` at lines 16-75 is what fabricates every score. Left in place as evidence — do not import it. |

### 3.2b `Optimized/` — corrected scoring and the model comparison

Added after the fabrication was found. Everything here is additive; nothing in
`SubFunctions/` or `mealpy/` was edited.

| File | What it does | How to run |
|---|---|---|
| **`metrics_fixed.py`** | Correct `Evaluation_Metrics` / `Evaluation_Metrics1`. Keeps the original formulas verbatim, including the class-0-as-positive convention; replaces only the confusion matrix with `sklearn`'s. Carries a self-test. | `python Optimized/metrics_fixed.py` → prints the self-test, exits non-zero on failure |
| **`optimize_models.py`** | The evaluation harness. Re-runs the paper's seven models with correct scoring, adds four current-generation backbones (EfficientNetV2-S, ConvNeXt-Tiny, MobileNetV3-Large, ResNet-RS-50) as frozen ImageNet extractors with trained heads, and an `SMA-CLMPNet-Opt` that keeps the published architecture but fixes the training recipe. Checkpoints after every split; `--resume` skips splits already on disk. | `python Optimized/optimize_models.py --mode sweep --resume --out Analysis1/TRUE`<br>`--mode kfold --ks 6,7,8,9,10`<br>`--mode diag --train-pct 0.6` |
| **`feature_probe.py`** | Answers whether the features carry any class signal at all: logistic regression / RBF SVM / random forest under repeated stratified CV across every feature representation, plus a 200-shuffle permutation test. | `python Optimized/feature_probe.py` |
| **`report.py`** | Builds `RESULTS.md` from `Analysis1/TP` (fabricated) and `Analysis1/TRUE` (measured), including the side-by-side gap table. | `python Optimized/report.py` |
| **`correct_doc.py`** | Rewrites §5.6.1, §5.6.2 and §5.8 of the `.docx` from the measured arrays, preserving each paragraph's formatting. Writes a timestamped backup. | `python Optimized/correct_doc.py [path.docx]` |
| `INTEGRITY_FINDING.md` | Evidence, blast radius and demonstration of the fabricated metric. | — |
| `RESULTS.md` | Generated results tables. | — |
| `cache/emb_*.npy` | Frozen backbone embeddings, computed once for all 50 samples. | — |

All of these need the env PATH set as in §2 and are run from the project root.

### 3.3 Data and output directories

| Path | Contents |
|---|---|
| `DATASET/` | FaceForensics++ videos. **Empty** — licensed, see §4. |
| `Features/Features.pkl` | Pre-extracted features. **Not in this repo** (1.0 GB > GitHub's 100 MB limit) — see §5. |
| `Analysis/` | The **published** result arrays. Read by `PlotResults`. Never overwritten by a re-run. |
| `Analysis1/TP`, `Analysis1/KF` | Where a **re-run writes** via `driver.py`. ⚠️ Scored by the tampered metric — not measurements. |
| `Analysis1/TRUE` | Measured results: all models, correct scoring. Columns are ACC, SEN, SPE, PRE, F1, **BAL-ACC**. |
| `Analysis1/TRUE_LATEST` | Measured results for the four current-generation backbones alone, all six splits. |
| `logs/sweep_true.log` | Live log of the corrected evaluation sweep. |
| `Results/` | Generated figures, CSVs and sample imagery. |
| `driver_out/` | Harness output: screenshots, evaluation tables, synthesised sample clip. |
| `RUN_REPORT.md` | Full narrative record of the reproduction run. |

---

## 4. Getting the dataset

FaceForensics++ has **no public download**. Access is granted only after the
authors approve a request, at which point they email a link to their download
script — the script is deliberately not in their public repository.

1. Submit the form: <https://github.com/ondyari/FaceForensics/tree/master/dataset>
2. Once approved, download only what this code reads:
   ```powershell
   python <their_download_script>.py DATASET -d FaceSwap  -c c23 -t videos
   python <their_download_script>.py DATASET -d original  -c c23 -t videos
   ```

The layout must be exactly:

```
DATASET/manipulated_sequences/FaceSwap/c23/videos/*.mp4   -> label 1 (forged)
DATASET/original_sequences/youtube/c23/videos/*.mp4       -> label 0 (authentic)
```

Deepfakes, Face2Face and NeuralTextures are **not** used by this code, nor are
the raw/c40 compression levels.

## 5. Getting `Features.pkl`

**You do not need the dataset to reproduce the evaluation.**
`Features/Features.pkl` already holds the features extracted from
FaceForensics++, and `ReadDataset(exec=False)` loads them. The raw videos are
needed only to *re-extract* features.

The file is 1.0 GB, over GitHub's hard 100 MB per-file limit, so it is
`.gitignore`d. Either copy it in from the original working directory, or
regenerate it once you have `DATASET/` populated:

```python
from SubFunctions import ReadDataset
ReadDataset(exec=True).read_data()      # writes Features/Features.pkl
```

Its contents:

| Key | Shape | Meaning |
|---|---|---|
| `comparative1/2/3/5` | (50, 128, 128, 10) | inputs to the comparison methods |
| `comparative4` | (50, 10, 12) | GLCM-style features |
| `proposed` | (50, 10, 128, 128, 12) | input to SMA-CLMPNet |
| `labels` | (50,) | **29 authentic (0) / 21 forged (1)** |

Note the corpus is **50 videos and is not class-balanced**.

---

## 6. Step-by-step: how the execution was performed

Run everything from the project root. `driver.py` `chdir`s to the project root
itself, because `Temp/themes/rose.json`, `Analysis/TP/*.npy` and the vendored
`./mealpy` are all resolved relative to the working directory.

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd <repo root>
$d = ".claude\skills\run-video-forgery-paper1\driver.py"
```

### Step 1 — verify the environment

```powershell
& "$E\python.exe" -u $d check
```

Imports every dependency, confirms the vendored `mealpy`, and reports the keras
weight cache. It calls `scipy.linalg.inv` **first**, as a canary for the
`Library\bin` DLL crash described in §2. Expect `CHECK OK`.

### Step 2 — synthesise a clip for the GUI

```powershell
& "$E\python.exe" -u $d make-video
```

The repository ships no video, but `GUI.py`'s first button demands one. This
builds `driver_out/sample.mp4` from the sample frames in
`Results/ImageResults/Input/`.

### Step 3 — drive the GUI and capture screenshots

```powershell
& "$E\python.exe" -u $d gui
```

Constructs the real `App()`, monkey-patches the modal file dialog, then calls
each handler in order, pumping Tk events and screenshotting after each into
`driver_out/screenshots/`:

| Screenshot | Stage | `App` method |
|---|---|---|
| `01-launched.png` | window as opened | — |
| `02-select.png` | Select Video | `select_data_event` |
| `03-preprocess.png` | Preprocessing (Haar face ROI) | `preprocessing_event` |
| `04-gradcam.png` | Grad-CAM heat-map | `get_gradcam` |
| `05-resnet.png` | ResNet + mean/var/std/skew/kurtosis | `get_resnetstat` |
| `06-vgg.png` | VGG LDZP | `get_vgg` |
| `07-flow.png` | Lucas-Kanade optical flow | `get_flow` |

Useful flags: `--stages select,preprocess,gradcam`, `--video path\to\clip.mp4`.

### Step 4 — regenerate all published figures

```powershell
& "$E\python.exe" -u $d plots
```

`Main.py`'s "No" branch without the modal prompt: forces the matplotlib **Agg**
backend and calls `PlotResults(show=False, save=True)`, so figures are written
instead of blocking. Produces **41 figures** plus CSVs under `Results/`.

### Step 5 — train and evaluate

```powershell
# single split, all models
& "$E\python.exe" -u $d evaluate --epochs 10

# training-percentage sweep, 40-90 %  (mirrors TPAnalysis.ComparativeAnalysis)
& "$E\python.exe" -u $d evaluate --sweep  --epochs 10 --skip BA-TFD

# K-fold, k = 6..10             (mirrors KFAnalysis.ComparativeAnalysis)
& "$E\python.exe" -u $d evaluate --kfold  --epochs 10 --folds-per-k 2 --skip BA-TFD

# learning curve for the proposed model
& "$E\python.exe" -u $d evaluate --curve 2,5,10,20,40 --train-pct 0.6
```

These use the same `train_test_split`, the same `Network` models and the same
`Evaluation_Metrics` as the original `Analysis.py`. Results are written to
`Analysis1/TP/` and `Analysis1/KF/` in the project's own `.npy` format (one row
per configuration, `[ACC, SEN, SPE, PRE, F1]`), plus readable tables in
`driver_out/`.

`Analysis1/` is deliberately separate from `Analysis/`, so **re-running never
overwrites the published arrays**.

Measured results and their caveats are in **[`RUN_REPORT.md`](RUN_REPORT.md)**.

---

## 7. Gotchas — things that will waste your day

- **`import skimage` hard-crashes without `Library\bin` on `PATH`** (§2). No
  traceback, exit `-1066598273`.

- **A bare import downloads 733 MB of weights.** `GetFeatures.py:19-20` and
  `GUI.py:23-24` call `ResNet101()` and `VGG16()` **at module scope**, so simply
  importing the package triggers the download (ResNet-101 180 MB, VGG-16 553 MB
  → `~/.keras/models`), with a progress bar that floods stdout. `get_gradcam`
  additionally pulls MobileNetV2 (14 MB) on first use.

- **`BA-TFD` (`Network.ViTDCNN`) cannot run — it OOMs at any batch size.** Its
  three `MaxPooling2D(1, 1)` layers are no-ops (pool size and stride both 1), so
  nothing is downsampled and `Flatten()` emits 128×128×64 = 1,048,576 features
  straight into `Dense(2048)`. That single weight matrix is `[1048576, 2048]` =
  **8.6 GB in float32**, before Adam's two moment copies and the gradient.
  Observed: `ResourceExhaustedError: OOM when allocating tensor with
  shape[1048576,2048]` with 22.5 GB free; identical at batch size 32, 2 and 1,
  because the weight matrix does not depend on batch size. Use `--skip BA-TFD`.
  Calling `TPAnalysis.ComparativeAnalysis()` unmodified aborts on this model.

- **`KFAnalysis.train_test_split` references a key that never exists.**
  `Analysis.py:355` loops `range(len(data['image']))`, but `GetData.py` only ever
  stores `comparative1-5`, `proposed` and `labels` — so `KFAnalysis` raises
  `KeyError: 'image'` on the shipped pickle. The driver indexes on
  `len(data['labels'])` instead, which is the same length and clearly the intent.

- **Tk never repaints unless you pump it.** The driver never calls `mainloop()`,
  so without manual `update_idletasks()` / `update()` loops every screenshot is a
  blank window.

- **Screenshots: neither Tk coordinates nor a DPI ratio give the right
  rectangle.** `winfo_rootx/rooty/width/height` are Tk-space, `PIL.ImageGrab` is
  physical pixels, and customtkinter applies a second scaling factor of its own.
  What works: `SetProcessDpiAwareness(2)` **before** tkinter is imported, plus
  Win32 `GetWindowRect` on the toplevel HWND via
  `GetAncestor(root.winfo_id(), GA_ROOT)` — `winfo_id()` is a child window.

- **`GUI.exit_event` is a `@staticmethod` referencing a module global `app`**
  that only exists under `if __name__ == "__main__"`. The driver assigns
  `GUI.app = app` after construction, or Exit raises `NameError`.

- **`GUI.refresh_event` needs `get_flow` to have run** — it calls
  `self.Home_12_inside.grid_forget()`, and that attribute is only created there.

- **`SubFunctions` prints emoji via termcolor**, raising `UnicodeEncodeError` on
  a cp1252 console and killing the run partway. The driver reconfigures stdout to
  UTF-8 with `errors="replace"`.

- **In PowerShell, `$?` lies about native exit codes.** `python -c "import
  seaborn"` returns 0 but writes a warning to stderr, and `$?` goes `False`.
  Check `$LASTEXITCODE`.

---

## 8. Provenance

The research code is **unmodified**. Every `.py` outside `.claude/` is
byte-identical to the original `CODE_28-04-2025(Paper1)` delivery. Files added by
the harness: `.claude/skills/run-video-forgery-paper1/`, `RUN_REPORT.md`,
`README.md`, `.gitignore` and `DATASET/README_PUT_VIDEOS_HERE.md`. The authors'
original README is preserved as `README_ORIGINAL.md`.
