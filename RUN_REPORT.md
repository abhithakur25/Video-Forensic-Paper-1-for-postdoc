# Paper 1 — implementation run report

Generated 2026-08-04 on Windows 11, conda env `VideoForgeryCPU` (Python 3.8.20),
CPU only (31.6 GB RAM, no CUDA).

Project: **"Design and development of video forgery model using deep learning
with attention mechanisms"** (SMA-CLMPNet). This folder is a self-contained copy
of `CODE_28-04-2025(Paper1)` plus the run-skill and the outputs below.

---

## 1. Dataset — NOT downloaded (blocked)

**FaceForensics++ cannot be downloaded programmatically.** There is no public
URL. Access is granted only after submitting the authors' Google form and being
approved, at which point they email a link to their download script; the script
is deliberately not in the public GitHub repo. Completing that request needs your
identity, institution and acceptance of their terms, so it was left for you.

Verified this session (both reachable, both confirm the gate):
- <https://www.niessnerlab.org/projects/roessler2019faceforensicspp.html>
- <https://github.com/ondyari/FaceForensics/tree/master/dataset>

`DATASET/` has been created with the exact layout the code globs, and
`DATASET/README_PUT_VIDEOS_HERE.md` carries the form link and the two download
commands. Drop the videos in and nothing else changes.

**The evaluation did not need it.** `Features/Features.pkl` (1.0 GB) already
contains features extracted from FaceForensics++, and
`ReadDataset(exec=False)` loads them. The raw videos are required only to
*re-extract* features (`Main.py`'s "Yes" branch, `exec=True`).

---

## 2. What was run

Every command below was executed in this folder and succeeded.

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd "C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1"
$d = ".claude\skills\run-video-forgery-paper1\driver.py"

& "$E\python.exe" -u $d check           # env + imports + weight cache      -> CHECK OK
& "$E\python.exe" -u $d make-video      # synthesises driver_out\sample.mp4 -> 318,084 bytes
& "$E\python.exe" -u $d plots           # 41 figures under Results\         -> 146.8 s
& "$E\python.exe" -u $d gui             # 6-stage GUI walkthrough           -> 7 screenshots
& "$E\python.exe" -u $d evaluate --sweep --epochs 10 --skip BA-TFD   # 5,639 s
```

| Output | Location |
|---|---|
| Analysis figures (41 PNG + CSV) | `Results/TP/`, `Results/KF/`, `Results/RocAnalysis/` |
| GUI screenshots (7 PNG) | `driver_out/screenshots/` |
| Evaluation tables | `driver_out/evaluation_sweep_ep10.txt`, `evaluation_tp90_ep*.txt` |
| Fresh comparative results | `Analysis1/TP/COM_A..H.npy` |
| Synthetic GUI input clip | `driver_out/sample.mp4` |

`Analysis1/` is where a re-run writes; the published `Analysis/` arrays were
**not** overwritten, so the paper's figures remain reproducible from source.

---

## 3. Evaluation results

Comparative analysis over training percentage 40–90 %, mirroring
`TPAnalysis.ComparativeAnalysis` — same `train_test_split`, same `Network`
models, same `Evaluation_Metrics`. **10 epochs, not the paper's 500.**

### Accuracy

| Model | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|
| EfficientNet | 0.8710 | 0.8846 | 0.9048 | 0.9375 | 0.9091 | 0.8333 |
| STIDNet | 1.0000 | 0.9231 | 0.7619 | 0.8750 | 0.7273 | 0.8333 |
| DCNN | 0.8710 | 0.8846 | 0.9048 | 1.0000 | 1.0000 | 0.8333 |
| GLCM | 0.9032 | 1.0000 | 0.8571 | 0.9375 | 1.0000 | 0.8333 |
| BA-TFD | — | — | — | — | — | — |
| MUSE-CLMPNet | 0.9677 | 0.9231 | 0.9524 | 0.8125 | 0.7273 | 1.0000 |
| SCAM-CLMPNet | 0.9355 | 0.9231 | 0.9524 | 0.8750 | 0.9091 | 0.8333 |
| **SMA-CLMPNet** (proposed) | 0.8710 | 0.8462 | 0.8095 | 1.0000 | 1.0000 | 1.0000 |

Test-set sizes: 31, 26, 21, 16, 11, **6** videos respectively.

### Across splits, vs the published numbers

| Model | mean acc | sd | min | max | paper @ 90% |
|---|---|---|---|---|---|
| EfficientNet | 0.8900 | 0.0328 | 0.8333 | 0.9375 | 0.9504 |
| STIDNet | 0.8534 | 0.0926 | 0.7273 | 1.0000 | 0.9507 |
| DCNN | 0.9156 | 0.0634 | 0.8333 | 1.0000 | 0.9600 |
| GLCM | 0.9219 | 0.0643 | 0.8333 | 1.0000 | 0.9615 |
| BA-TFD | not run | | | | 0.9629 |
| MUSE-CLMPNet | 0.8972 | 0.0960 | 0.7273 | 1.0000 | 0.9685 |
| SCAM-CLMPNet | 0.9047 | 0.0399 | 0.8333 | 0.9524 | 0.9717 |
| **SMA-CLMPNet** | 0.9211 | 0.0809 | 0.8095 | 1.0000 | 0.9792 |

### How to read this

**This is a smoke-scale run, not a reproduction, and it neither confirms nor
refutes the paper.** Three reasons to be careful:

1. **10 epochs vs 500.** The proposed model is the deepest here and is the one
   most penalised by early stopping — at 40–60 % training it is the *worst*
   model in this run, then ties for best at 70–90 %. That pattern is what
   undertraining looks like.
2. **The test sets are tiny.** At 90 % training the test set is 6 videos, so the
   only achievable accuracies are 0.0, 0.1667, 0.3333, … Every 90 % column value
   is one of 0.8333 (5/6) or 1.0000 (6/6). Differences there carry no
   information.
3. **No monotonic trend.** The paper reports accuracy rising smoothly with
   training percentage; at 10 epochs the curves are dominated by split noise.

To attempt a real reproduction, run `--epochs 500` (est. ~24 h, see §5).

---

## 4. Two findings about the code

### 4.1 BA-TFD cannot execute — it OOMs at any batch size

`Network.ViTDCNN` (`SubFunctions/Model.py:377`) uses three
`MaxPooling2D(1, 1)` layers. Pool size and stride are both 1, so they are
**no-ops** — nothing is downsampled. `Flatten()` therefore emits
128 × 128 × 64 = **1,048,576** features directly into `Dense(2048)`, giving a
single weight matrix of `[1048576, 2048]` = **8.6 GB in float32**, before Adam's
two moment copies and the gradient (~34 GB total).

Observed: `ResourceExhaustedError: OOM when allocating tensor with
shape[1048576,2048]` with 22.5 GB free. Retried at `batch_size` 32, 2 and 1 —
identical failure, because the weight matrix does not depend on batch size.

Consequence: calling `TPAnalysis.ComparativeAnalysis()` unmodified aborts here.
The driver's `--skip BA-TFD` works around it. If the pooling was meant to be
`MaxPooling2D(2, 2)`, that is a one-character-per-layer fix — but it changes the
published architecture, so it was **not** applied.

### 4.2 The feature set is 50 videos, unbalanced

`Features/Features.pkl` contains:

| key | shape | meaning |
|---|---|---|
| `comparative1/2/3/5` | (50, 128, 128, 10) | inputs for the comparison methods |
| `comparative4` | (50, 10, 12) | GLCM-style features |
| `proposed` | (50, 10, 128, 128, 12) | input for SMA-CLMPNet |
| `labels` | (50,) | **29 authentic (0) / 21 forged (1)** |

So the experiments behind the paper rest on **50 videos**, not the 1000 the
dataset description implies, and the two classes are **not** equal (29 vs 21) —
which sits awkwardly against the paper's statement that the dataset "contains two
classes Normal and Scam with an equal number of counts". Worth reconciling before
submission, independently of anything in this run.

---

## 5. To finish the job

1. Request FaceForensics++ access (§1), download FaceSwap + originals at c23 into
   `DATASET/`.
2. Full-scale evaluation on the shipped features — no dataset needed:
   ```powershell
   & "$E\python.exe" -u $d evaluate --sweep --epochs 500 --skip BA-TFD
   ```
   Estimated ~24 h: SMA-CLMPNet alone measured **~29 s/epoch**, and it is one of
   seven models across six splits. This matches the README's 48 h figure.
3. Re-extract features from the real videos (`ReadDataset(exec=True)`) only if you
   want a larger corpus than the 50 videos in the pickle — note the result will
   differ from the published numbers.
4. Decide what to do about BA-TFD (§4.1) — fix the pooling, or drop it from the
   comparison and say so.

---

## 6. GitHub repository

Pushed to **<https://github.com/abhithakur25/Video-Forensic-Paper-1-for-postdoc>**
(private), branch `main`, 1672 files.

`Features/Features.pkl` is excluded by `.gitignore` — at 1.0 GB it exceeds
GitHub's hard 100 MB per-file limit, and a 1 GB file does not fit the free Git
LFS quota either. `README.md` §5 documents how to copy or regenerate it. Nothing
else is excluded except the licensed `DATASET/` videos and the usual Python
noise.

The full project description — environment setup, a table covering every
executable file, the step-by-step execution sequence and the gotchas — is in
`README.md` at the repository root. The authors' original README is preserved
verbatim as `README_ORIGINAL.md`.

---

## 7. Provenance

Verified: **no research source was modified.** Every `.py` outside `.claude/` is
byte-identical to `CODE_28-04-2025(Paper1)`. Differences from the original:

| File | Status |
|---|---|
| `.claude/skills/run-video-forgery-paper1/driver.py` | modified — added `evaluate` / `--sweep` |
| `.claude/skills/run-video-forgery-paper1/SKILL.md` | modified — documented the above + gotchas |
| `DATASET/README_PUT_VIDEOS_HERE.md` | new |
| `Analysis1/TP/COM_A..H.npy` | regenerated by this run |
| `Results/**/*.png`, `driver_out/**` | regenerated by this run |
