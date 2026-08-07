# Resume state — 2026-08-08 04:36

Written because execution was stopped mid-run. Everything below is either on
disk or reproducible from a committed script. Nothing here is a plan; it is
what exists right now.

## 1. FF++ corpus — COMPLETE, verified, ready to upload

`DATASET/ffpp_frames.zip`, **1.00 GB**, built from 2,000 videos.

```
63,971 face crops, 299 px, identity-level split
  train  23,035 real / 23,032 fake
  val     4,477 real /  4,467 fake
  test    4,480 real /  4,480 fake
```

Verification that passed (`Optimized/repack_ffpp.py`, re-runnable with
`--verify-only`):

- 63,981 archive entries, **0 containing a backslash**
- 63,971 manifest rows, **0 backslash paths, 0 rows without a matching entry**
- `train/`, `val/`, `test/`, `manifest.csv` at the archive root
- **no identity appears in two splits**

**Next action, needs a human:** upload to Google Drive as
`MyDrive/ffpp_frames.zip`, then run `FFPP_Colab_Train.ipynb` on a T4.

Do not rebuild the archive with PowerShell's
`[System.IO.Compression.ZipFile]::CreateFromDirectory` — on .NET Framework it
writes backslash entry names and the result is silently unusable on Linux. Use
`Optimized/repack_ffpp.py`.

## 2. SMA-CLMPNet recipe search — INTERRUPTED mid fold 2

Nested 5 outer x 2 inner over 6 configurations, 65 fits total.
Log: `logs/smaclmpnet_search.log`  Checkpoint: `Optimized/smaclmpnet_search.json`

### Checkpointed (survives restart)

| Outer fold | Chosen configuration | Inner | **Outer (honest)** | Time |
|---|---|---|---|---|
| 1 | batch 4, 30 ep, lr 1e-3 flat, normalise, no class weight | 54.17 | **50.00** | 265.9 min |

### NOT checkpointed — will be recomputed on resume

Fold 2 configs 1-4 all scored **50.00** inner (~2.2 h of compute). The
checkpoint is written per *outer fold*, so an interrupted fold is lost. A
partial snapshot of the log is kept at
`logs/smaclmpnet_partial_20260808-043627.log`.

### Every fit measured so far

```
fold 1 cfg 1/6  50.00   batch 32, 10 ep, 1e-3 flat            <- PUBLISHED recipe
fold 1 cfg 2/6  50.00   batch  8, 30 ep, 1e-3 cosine, cw, norm <- SMA-CLMPNet-Opt
fold 1 cfg 3/6  52.08   batch 16, 60 ep, 1e-3 flat, cw, norm
fold 1 cfg 4/6  54.17   batch  4, 30 ep, 1e-3 flat, norm
fold 1 cfg 5/6  50.00   batch  4, 30 ep, 3e-4 flat, cw, norm
fold 1 cfg 6/6  46.88   batch  4, 60 ep, 3e-3 flat, cw
fold 1 OUTER    50.00
fold 2 cfg 1/6  50.00
fold 2 cfg 2/6  50.00
fold 2 cfg 3/6  50.00
fold 2 cfg 4/6  50.00
```

### What this already shows

Both published anchors score exactly 50.00 — on a 29/21 corpus that is
single-class prediction. Inner scores range 46.88 to 54.17, which is noise
around chance: one sample in a 24-sample inner fold moves balanced accuracy by
about 2 points, so the whole observed spread is roughly two samples wide.

Fold 1 is the pattern in miniature: the best inner score was 54.17 and the same
configuration scored **50.00** on the held-out outer fold. That 4.17-point drop
is selection bias, measured rather than argued, and it is exactly why the search
is nested. A non-nested search would have reported 54.17.

**Nothing here beats L1 logistic regression on temporal deltas at 77.17 %.**
With 2,258,534 parameters against 40 training samples, the constraint is sample
count, not the recipe.

### To resume

```bat
REM detached, WMI-owned, picks up from the fold-1 checkpoint
Optimized\run_smaclmpnet_search.bat
```

Launch it through `Invoke-CimMethod -ClassName Win32_Process -MethodName Create`,
not `Start-Process` — the latter does not escape the harness job object and the
process dies with the session. Roughly 14 h of work remain (folds 2-5 at ~4.4 h
each, minus the ~2.2 h of fold 2 that must be redone).

**Worth fixing before a long re-run:** checkpoint after each *config* rather
than each outer fold. As written, an interruption costs up to 4.4 h.

## 3. Documents — regenerated 2026-08-07 21:55

`Research_Paper-1.docx`, `Paper1_Complete_Work_Report.docx` and
`Results/results_dashboard.html` were rebuilt after the §5.11.1 pairing fix.
Backups at `*.bak-20260807-215521` (git-ignored).

The leakage sentence now reads `57.64 % -> 85.54 %` for Xception RGB frames,
both halves the same representation. Remaining `58.95` occurrences are correct:
they are the measured video-grouped score for Xception RGB + differences in
TABLE 15/16 and in a separate best-vs-best sentence that makes no same-features
claim.

## 4. Not mine, was running

Two `python -m kaggle datasets download` processes were pulling
`manjilkarki/deepfake-and-real-images` and
`saurabhbagchi/deepfake-image-detection` into
`C:\Users\USER\Downloads\kaggle_run\input\datasets\`. Free space was 45.9 GB.
`archive.zip` (16.7 GB) in the parent directory is now redundant — the corpus is
extracted and the verified archive is built — if space is needed.

## 5. Environment reminders

- Only `C:\Users\USER\anaconda3\envs\VideoForgeryCPU` (Python 3.8.20) runs this
  code, and `<env>\Library\bin` must be on PATH or `import skimage` hard-crashes.
- torch is unavailable; the TLS-intercepting proxy breaks conda installs.
- Long jobs must be launched via WMI to survive the session.
- Commit messages with embedded quotes break PowerShell here-strings; use
  `git commit -F <file>`.
