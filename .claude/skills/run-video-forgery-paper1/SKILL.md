---
name: run-video-forgery-paper1
description: Build, run, and drive the Paper 1 video forgery detection project (deep learning with attention mechanisms, LDZP + optical flow). Use when asked to start or launch this app, run the GUI, regenerate the analysis plots/figures, take a screenshot of the UI, verify a change works in the real app, or smoke-test the feature-extraction pipeline.
---

Paper 1 ("Design and development of video forgery model using deep learning with
attention mechanisms") is a Windows Python 3.8 research codebase with two surfaces:
a customtkinter desktop GUI (`GUI.py`) and a plot-regeneration script (`Main.py`).
**Both block on modal prompts and cannot be driven from a shell.** Drive them with
`.claude/skills/run-video-forgery-paper1/driver.py`, which bypasses the prompts,
walks the GUI stage by stage, and writes PNG screenshots.

All paths below are relative to `CODE_28-04-2025(Paper1)/`.
This is Windows + PowerShell, not Linux — there is no xvfb and none is needed;
the GUI renders on the real desktop.

Paper 2 (`CODE_05-08-2025(Paper2)/`) is a near-identical codebase with its own
skill; the same driver serves both. Paper 1 differs in its feature stages
(VGG LDZP and Lucas-Kanade optical flow instead of SIFT-VGG16 and shape-ResNet)
and, usefully, **never imports torch**.

## Prerequisites

Use the existing conda env `VideoForgeryCPU` (Python 3.8.20) — it already matches
`requirements.txt`. Every command below assumes these two lines first:

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
```

**The `$E\Library\bin` entry is mandatory, not cosmetic** — see Gotchas.
`conda activate VideoForgeryCPU` sets the same thing and works too.

The interpreter is `$E\python.exe`. Do **not** use the `python` on PATH: that is
3.14.6, and neither TensorFlow 2.10 nor numpy 1.21.6 exists for it.

### Rebuilding the env from scratch

> Unlike everything else in this file, this block was **not executed** in the
> session that wrote it — this network's TLS interception broke both `pip` and
> `conda` downloads partway through (see Troubleshooting). It is reconstructed
> from the versions actually present in the working env, so treat the pins as
> verified and the commands as untested.

```powershell
conda create -n VideoForgeryCPU python=3.8 -y
conda activate VideoForgeryCPU
conda install -y numpy=1.21.6 scipy=1.7.3 pandas=1.3.4 matplotlib=3.5.3 `
    seaborn=0.12.2 scikit-learn=1.0.2 scikit-image=0.19.3 -c conda-forge
pip install tensorflow==2.10.0 keras==2.10.0 opencv-python==4.8.0.76 `
    Pillow==9.5.0 termcolor tqdm customtkinter==5.1.3 "PySimpleGUI==4.60.5.1"
```

Deviations from `requirements.txt` that are required, not optional:

- `PySimpleGUI` is imported by `Main.py` but **missing from
  `requirements.txt` entirely**; and `4.60.5` (the pin Paper 2 uses) no longer
  installs — it was pulled from PyPI when the project relicensed. Use `4.60.5.1`,
  the last free 4.x. It provides `popup_yes_no`, which is all `Main.py` uses.
- `pip install customtkinter` (unpinned) gives 6.0.0, whose API differs. Pin `5.1.3`.
- `scikit-image` must stay on 0.19.x: the code calls `greycomatrix`/`greycoprops`
  (`SubFunctions/GetFeatures.py:22`), renamed to `gray*` in 0.20 and removed later.

## Run (agent path)

Everything goes through the driver. Run it from the project root:

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd "C:\Users\USER\Downloads\PostDoc\CODE_28-04-2025(Paper1)"
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper1\driver.py" check
```

Always pass `-u`. Without it a hard crash (see Gotchas) discards buffered stdout
and the failure looks like silence.

`check` verifies the interpreter, every import, the vendored `mealpy`, and the
keras weight cache. Expected tail:

```
[driver]   torch UNAVAILABLE (OSError) - this project never imports torch, so it does not matter
[driver]   vendored ./mealpy present
[driver]   cached resnet101_weights_tf_dim_ordering_tf_kernels.h5 (180 MB)
[driver]   cached vgg16_weights_tf_dim_ordering_tf_kernels.h5 (553 MB)
[driver] CHECK OK
```

### Drive the GUI and take screenshots

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper1\driver.py" make-video
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper1\driver.py" gui
```

`make-video` is required once — the repo ships no `DATASET/` and no video, and the
GUI's first button needs one. It builds `driver_out/sample.mp4` from the sample
frames in `Results/ImageResults/Input/`.

`gui` builds the real `App()`, then calls each handler in order, pumping Tk events
and screenshotting after each. Screenshots land in `driver_out/screenshots/`:

| file | stage | `App` method |
|---|---|---|
| `01-launched.png` | window as opened | — |
| `02-select.png` | Select Video | `select_data_event` |
| `03-preprocess.png` | Preprocessing (haar face ROI) | `preprocessing_event` |
| `04-gradcam.png` | GradCAM heatmap | `get_gradcam` |
| `05-resnet.png` | ResNet + mean/var/std/skew/kurtosis | `get_resnetstat` |
| `06-vgg.png` | VGG LDZP | `get_vgg` |
| `07-flow.png` | Lucas-Kanade optical flow tracks | `get_flow` |

Whole run is ~25 s once weights are cached. Subsets: `--stages select,preprocess,gradcam`.
Another video: `--video path\to\clip.mp4`.

**Look at the PNGs afterwards.** A stage can "pass" and still render nothing.

### Train and evaluate (no raw videos needed)

```powershell
# one split, all models
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper1\driver.py" evaluate --epochs 10

# the paper's 40-90%% training-percentage sweep
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper1\driver.py" `
    evaluate --sweep --epochs 10 --skip BA-TFD
```

`Features/Features.pkl` holds features already extracted from FaceForensics++,
so `ReadDataset(exec=False)` gives you the real training/evaluation path with no
`DATASET/` present. This mirrors `TPAnalysis.ComparativeAnalysis`: same
`train_test_split`, same `Network` models, same `Evaluation_Metrics`.

`--sweep` writes `Analysis1/TP/COM_A..H.npy` in the project's own format (one row
per training percentage, `[ACC, SEN, SPE, PRE, F1]`) plus a readable table in
`driver_out/`. It writes to **`Analysis1/`, not `Analysis/`**, so the published
figures are never overwritten.

Scale honestly: the paper uses `epochs=500` across 6 training percentages.
Measured here, SMA-CLMPNet costs ~29 s/epoch, so the published configuration is
roughly a 24-hour run — consistent with the README's 48 h estimate. `--epochs 10`
finishes in ~1.5 h and is a smoke-scale result, not a reproduction.

`--skip BA-TFD` is required — see Gotchas.

### Regenerate the analysis figures

```powershell
& "$E\python.exe" -u ".claude\skills\run-video-forgery-paper1\driver.py" plots
```

This is `Main.py`'s "No" branch (plots from pre-evaluated `Analysis/*.npy`) with
the modal prompt removed and `matplotlib` forced to Agg. Writes **41** figures
plus CSVs under `Results/` (TP + KF comparative/performance bar and line graphs,
plus the ROC analysis). Timing varies a lot — observed 53 s and 114 s on the same
machine; allow a couple of minutes.

`check`, `make-video`, `plots` then `gui` in one go: `driver.py all`.

## Run (human path)

```powershell
& "$E\python.exe" GUI.py     # opens the window; drive it by hand, close to exit
& "$E\python.exe" Main.py    # modal Yes/No popup, then ~40 blocking plt.show() windows
```

Both are interactive-only and unusable for automation. `Main.py`'s "Yes" branch is
the full 48-hour training run and needs a `DATASET/` (FaceForensics++) that is not
in the repo.

## Gotchas

- **`import skimage` hard-crashes the interpreter if `$E\Library\bin` is not on
  PATH.** Exit code `-1066598273` (0xC0000409, "stack buffer overrun"), no
  traceback. `-X faulthandler` traces it to `skimage/color/colorconv.py:396` →
  `scipy.linalg.inv` → Windows exception `0xc06d007f`, a **DLL delay-load
  failure**: conda's MKL lives in `Library\bin`, and invoking `python.exe` by
  absolute path (rather than `conda activate`) leaves it off PATH. The driver's
  `check` calls `scipy.linalg.inv` first as a canary.

- **torch is uninstallable in this env — and irrelevant to this project.**
  torch and conda's MKL scipy both ship `libiomp5md.dll` (conda's is a 157 KB
  stub, torch's the real 1.9 MB Intel OpenMP) and clash whichever order they
  load. Paper 1 never imports torch, so `check` reporting it as unavailable is
  expected and harmless. (Paper 2 does need it for its full-analysis path.)

- **The driver never runs `SubFunctions/__init__.py`.** `__init__` does
  `from .Analysis import ...` → `Model` → `mealpy`/keras, which costs minutes on
  import, and `GUI.py` needs only `SubFunctions.GradCAM` and `SubFunctions.LDZP`.
  `driver.subfunctions_lite()` registers `SubFunctions` as a bare package (a
  module object with `__path__`) so submodule imports resolve without executing
  `__init__` — safe because `GradCAM.py`, `LDZP.py` and `VisualizeResults.py`
  import nothing else from the package. GUI import drops to ~10 s. The
  full-analysis path does need the real `__init__`; `--full-package` opts back in.

- **Importing this project downloads 733 MB of keras weights.**
  `SubFunctions/GetFeatures.py:19-20` and `GUI.py:23-24` call `ResNet101()` and
  `VGG16()` **at module scope**, so a bare `import` triggers it (resnet101 180 MB,
  vgg16 553 MB → `~/.keras/models`), with a progress bar that floods stdout.
  `get_gradcam` additionally pulls MobileNetV2 (14 MB) on first use. Cached after
  the first run; `driver.py check` reports which are present.

- **BA-TFD (`Network.ViTDCNN`) cannot run — it OOMs at any batch size.** Its
  three `MaxPooling2D(1, 1)` layers are no-ops (pool size and stride both 1), so
  nothing is downsampled and `Flatten()` emits 128x128x64 = 1,048,576 features
  straight into `Dense(2048)`. That single weight matrix is
  `[1048576, 2048]` = **8.6 GB in float32**, before Adam's two moment copies and
  the gradient. Measured failure: `ResourceExhaustedError: OOM when allocating
  tensor with shape[1048576,2048]` on a machine with 22.5 GB free. Lowering
  `batch_size` to 2 and then 1 does not help, because the weight matrix is
  independent of batch size. Pass `--skip BA-TFD`; calling the project's own
  `TPAnalysis.ComparativeAnalysis()` directly will abort on this model.

- **`Features/Features.pkl` contains only 50 videos (29 authentic / 21 forged),**
  not the 1000+ implied by the dataset description, and the classes are not
  balanced. With `--train-pct 0.9` the test set is **6 videos**, so single-split
  accuracies land on coarse fractions (0.8333 = 5/6) and 1.0000 is common. Use
  `--sweep`, and treat any single number with suspicion.

- **Screenshots: neither Tk coordinates nor a DPI ratio give the right rectangle.**
  `winfo_rootx/rooty/width/height` are Tk-space; `PIL.ImageGrab` is physical
  pixels; and customtkinter applies a *second* scaling factor of its own. Scaling
  the Tk box by screen-size ratio captured the desktop and taskbar instead of the
  window. What works: `SetProcessDpiAwareness(2)` **before** tkinter is imported,
  plus Win32 `GetWindowRect` on the toplevel HWND — reached via
  `GetAncestor(root.winfo_id(), GA_ROOT)`, since `winfo_id()` is a child window.

- **Tk never repaints unless you pump it.** The driver never calls `mainloop()`,
  so without manual `update_idletasks()`/`update()` loops every screenshot is a
  blank window.

- **`GUI.exit_event` is a `@staticmethod` referencing a module global `app`**
  that only exists under `if __name__ == "__main__"`. The driver assigns
  `GUI.app = app` after constructing it, or Exit raises `NameError`.

- **`GUI.refresh_event` calls `self.Home_12_inside.grid_forget()`**, and
  `Home_12_inside` is assigned only inside `get_flow` — so Refresh is only safe
  after the optical-flow stage has run. It is not in the driver's stage list.

- **`SubFunctions` prints emoji through termcolor**, which raises
  `UnicodeEncodeError` on a cp1252 console and kills the run partway. The driver
  reconfigures stdout to utf-8/replace.

- **`PlotResults()` defaults to `show=True`**, i.e. ~40 blocking `plt.show()`
  windows. `PlotResults(show=False, save=True)` writes files instead — that flag
  pair is the whole difference between the human and agent paths.

- **CWD must be the project root.** `Temp\themes\rose.json`, `Analysis\TP\*.npy`
  and the vendored `./mealpy` are all resolved relative to CWD. The driver
  `chdir`s itself, so it can be invoked from anywhere.

- **In PowerShell, `$?` lies about native exit codes.** `python -c "import seaborn"`
  returned 0 but wrote a warning to stderr, and `$?` went `False`. Check
  `$LASTEXITCODE`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Exit `-1066598273` or `3`, no traceback | `Library\bin` missing from PATH (skimage/scipy), or torch got imported. Re-run `driver.py check`. |
| `OSError [WinError 182] ... shm.dll` | torch vs conda MKL clash. Irrelevant here — Paper 1 does not use torch. |
| Silent failure, no output at all | Missing `-u`; buffered stdout is lost on a hard crash. |
| `ERROR: Could not find a version that satisfies the requirement PySimpleGUI==4.60.5` | Use `4.60.5.1`. |
| `ModuleNotFoundError: No module named 'PySimpleGUI'` | It is missing from `requirements.txt`; install it explicitly. |
| `ImportError: cannot import name 'greycomatrix'` | scikit-image ≥ 0.20. Pin `0.19.3`. |
| `pip`/`conda` → `CERTIFICATE_VERIFY_FAILED, self-signed certificate in chain` | TLS-intercepting proxy. Export the Windows trusted roots to a PEM and pass `pip --cert <pem>`; this fixed verification (downloads may still reset). |
| GUI screenshot shows desktop/taskbar instead of the window | DPI awareness not set before importing tkinter, or Tk coords used instead of `GetWindowRect`. |
| Screenshots blank | Tk events not pumped between actions. |
| `driver.py gui` → "no video at ...; run 'make-video' first" | Run the `make-video` command. |
| First run appears to hang with a progress bar | Downloading the 733 MB of keras weights. Let it finish once. |
