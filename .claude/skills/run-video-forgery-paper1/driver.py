#!/usr/bin/env python
"""
Agent-facing driver for the video-forgery-detection research code.

The app has two surfaces and neither can be driven from a plain shell:

  * GUI.py  - a customtkinter desktop app whose only entry point is
              app.mainloop(), and whose "Select Video" button opens a
              blocking native file dialog.
  * Main.py - blocks on a PySimpleGUI popup_yes_no(), then calls
              PlotResults() with show=True, which blocks on plt.show()
              once per figure (~40 figures).

This driver bypasses both blocking prompts and drives the real code.

Commands:
  check        env sanity: imports, keras weight cache, vendored mealpy
  make-video   synthesize driver_out/sample.mp4 from Results/ImageResults/Input
  plots        regenerate every analysis figure headlessly into Results/
  gui          walk the GUI through all 6 stages, screenshot each
  all          check -> make-video -> plots -> gui

Run it with the project directory as CWD, or from anywhere - it chdir's to
the project root itself (relative paths like "Temp\\themes\\rose.json" and
the vendored ./mealpy package require it).
"""
import argparse
import ctypes
import os
import sys
import time
import traceback
from pathlib import Path

# <unit>/.claude/skills/run-*/driver.py  ->  <unit>
PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "driver_out"
SHOTS = OUT / "screenshots"


def log(msg):
    sys.stdout.write(f"[driver] {msg}\n")
    sys.stdout.flush()


def subfunctions_lite():
    """Register `SubFunctions` as a bare package so that submodule imports
    work WITHOUT executing SubFunctions/__init__.py.

    __init__.py does `from .Analysis import ...` -> Model -> Attention -> torch,
    and in this conda env torch cannot be imported in the same process as the
    conda-forge MKL build of scipy (see SKILL.md "Gotchas"). Nothing the GUI or
    the plotting code needs lives behind that import: GUI.py only wants
    SubFunctions.GradCAM (+ SubFunctions.LDZP on Paper1) and the plots path only
    wants SubFunctions.VisualizeResults - none of which import anything else
    from the package. Skipping __init__ also avoids ~3 min of Analysis/Model
    /mealpy import time.

    The full 48-hour ReadDataset/TPAnalysis/KFAnalysis path DOES need the real
    __init__ (and therefore a working torch); use --full-package for that.
    """
    import types
    if "SubFunctions" in sys.modules:
        return
    pkg = types.ModuleType("SubFunctions")
    pkg.__path__ = [str(PROJECT / "SubFunctions")]
    pkg.__package__ = "SubFunctions"
    sys.modules["SubFunctions"] = pkg
    log("SubFunctions registered without running __init__.py (torch bypass)")


def setup():
    """chdir to project root and make stdout tolerate the code's emoji output."""
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    OUT.mkdir(exist_ok=True)
    SHOTS.mkdir(exist_ok=True)
    # SubFunctions prints emoji via termcolor; cp1252 consoles raise
    # UnicodeEncodeError and kill the run partway through.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def cmd_check(args):
    ok = True
    log(f"project root : {PROJECT}")
    log(f"python       : {sys.version.split()[0]} @ {sys.executable}")

    # scipy.linalg is the canary: if the conda env's Library\bin is missing
    # from PATH, this hard-crashes the interpreter (0xC0000409) instead of
    # raising, and takes skimage down with it.
    import numpy as np
    import scipy.linalg
    scipy.linalg.inv(np.eye(3))
    log("scipy.linalg LAPACK delay-load OK (Library\\bin is on PATH)")

    for mod in ("cv2", "tensorflow", "keras", "skimage", "customtkinter",
                "PySimpleGUI", "peakutils", "seaborn", "sklearn",
                "pandas", "tqdm", "PIL"):
        try:
            m = __import__(mod)
            log(f"  import {mod:<14} {getattr(m, '__version__', '?')}")
        except Exception as e:
            ok = False
            log(f"  IMPORT FAILED {mod}: {type(e).__name__}: {e}")

    # torch is expected to fail here: it cannot share a process with the
    # conda-forge MKL build of scipy. Not fatal - nothing this driver runs
    # needs it (and Paper1 never imports it at all).
    uses_torch = any(
        "import torch" in p.read_text(encoding="utf-8", errors="replace")
        for p in (PROJECT / "SubFunctions").glob("*.py"))
    try:
        import torch
        log(f"  import {'torch':<14} {torch.__version__}")
    except Exception as e:
        if uses_torch:
            log(f"  torch UNAVAILABLE ({type(e).__name__}) - expected in this "
                f"env; only the full-analysis path needs it")
        else:
            log(f"  torch UNAVAILABLE ({type(e).__name__}) - this project never "
                f"imports torch, so it does not matter")

    from skimage.feature import greycomatrix  # noqa: F401  (0.19.x spelling)
    log("  skimage.feature.greycomatrix present (needs scikit-image 0.19.x)")

    if not (PROJECT / "mealpy" / "__init__.py").exists():
        ok = False
        log("  MISSING vendored ./mealpy")
    else:
        log("  vendored ./mealpy present")

    cache = Path.home() / ".keras" / "models"
    for w in ("resnet101_weights_tf_dim_ordering_tf_kernels.h5",
              "vgg16_weights_tf_dim_ordering_tf_kernels.h5"):
        p = cache / w
        if p.exists():
            log(f"  cached {w} ({p.stat().st_size / 1e6:.0f} MB)")
        else:
            log(f"  NOT CACHED {w} - first GUI/import run will download it")

    log("CHECK OK" if ok else "CHECK FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# make-video
# --------------------------------------------------------------------------
def cmd_make_video(args):
    """The repo ships no DATASET/ and no video, but GUI.select_data_event
    needs one. Build a short clip from the sample frames in Results/."""
    import cv2
    import numpy as np

    dst = OUT / "sample.mp4"
    src = sorted((PROJECT / "Results" / "ImageResults" / "Input").glob("*.jpg"))
    frames = []
    for p in src[:30]:
        im = cv2.imread(str(p))
        if im is not None:
            frames.append(cv2.resize(im, (256, 256)))
    if not frames:
        log("no sample frames found; generating synthetic ones")
        for i in range(30):
            f = np.full((256, 256, 3), 40, np.uint8)
            cv2.circle(f, (60 + 4 * i, 128), 40, (200, 180, 160), -1)
            frames.append(f)

    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 10, (256, 256))
    for f in frames:
        vw.write(f)
    vw.release()
    log(f"wrote {dst} ({len(frames)} frames, {dst.stat().st_size} bytes)")
    return 0


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def cmd_plots(args):
    """Main.py's else-branch, minus the two blocking prompts.

    Main.py does PlotResults() -> show=True -> plt.show() per figure. Under
    Agg with show=False/save=True the same figures land in Results/ instead.
    """
    import matplotlib
    matplotlib.use("Agg")

    if getattr(args, "full_package", False):
        from SubFunctions import PlotResults
    else:
        subfunctions_lite()
        from SubFunctions.VisualizeResults import PlotResults

    before = {p: p.stat().st_mtime for p in PROJECT.glob("Results/**/*.png")}
    pl = PlotResults(show=False, save=True)
    t0 = time.time()
    pl.TPAnalysisResult()
    pl.KFAnalysisResult()
    dt = time.time() - t0

    after = list(PROJECT.glob("Results/**/*.png"))
    fresh = [p for p in after if p not in before or p.stat().st_mtime > before[p]]
    log(f"plots done in {dt:.1f}s - {len(fresh)} figures written/updated "
        f"({len(after)} PNGs total under Results/)")
    for p in sorted(fresh)[:8]:
        log(f"    {p.relative_to(PROJECT)}")
    if len(fresh) > 8:
        log(f"    ... and {len(fresh) - 8} more")
    return 0 if fresh else 1


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------
MODELS = [
    ("EfficientNet", lambda n: n.EfficientNet()),
    ("STIDNet", lambda n: n.STIDNet()),
    ("DCNN", lambda n: n.CNN()),
    ("GLCM", lambda n: n.GLCM()),
    ("BA-TFD", lambda n: n.ViTDCNN()),
    ("MUSE-CLMPNet", lambda n: n.ThreeDCNNLSTM(opt=1)),
    ("SCAM-CLMPNet", lambda n: n.ThreeDCNNLSTM(opt=2)),
    ("SMA-CLMPNet", lambda n: n.ThreeDCNNLSTM(opt=3)),
]


def _sweep(args, data, lab):
    """Replicate TPAnalysis.ComparativeAnalysis: training percentage 0.4..0.9,
    every model, results saved as Analysis1/TP/COM_<A..H>.npy in the project's
    own format (one row per training percentage, [ACC,SEN,SPE,PRE,F1]).

    We drive the loop here rather than calling ComparativeAnalysis() directly
    because that method also runs BA-TFD, which OOMs (see cmd_evaluate notes)
    and would abort the whole sweep. Everything else is identical.
    """
    import numpy as np
    from SubFunctions.Analysis import train_test_split
    from SubFunctions.Model import Network
    from SubFunctions.Evaluate import Evaluation_Metrics

    skip = set((args.skip or "").split(",")) - {""}
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    pcts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    grid = {name: [] for name, _ in MODELS}
    t_all = time.time()

    for si, tp in enumerate(pcts):
        d = train_test_split(data, train_size=tp)
        params = {f"x_train{i+1}": d[i] for i in range(6)}
        params.update({f"x_test{i+1}": d[6 + i] for i in range(6)})
        params.update({"y_train": d[12], "y_test": d[13], "epochs": args.epochs})
        y_test = np.asarray(d[13])
        log(f"===== split {si+1}/6  train_size={tp}  "
            f"(train n={len(np.asarray(d[12]))}, test n={len(y_test)}) =====")
        for name, fn in MODELS:
            if name in skip:
                grid[name].append([np.nan] * 5)
                continue
            t = time.time()
            try:
                net = Network(**params)
                if args.batch_size:
                    net.batch_size = args.batch_size
                m = Evaluation_Metrics(y_test, np.asarray(fn(net)))
                grid[name].append(list(m))
                log(f"  {name:<14} acc={m[0]:.4f} sens={m[1]:.4f} spec={m[2]:.4f} "
                    f"prec={m[3]:.4f} f1={m[4]:.4f}  ({time.time()-t:.1f}s)")
            except Exception as e:
                grid[name].append([np.nan] * 5)
                log(f"  {name:<14} FAILED {type(e).__name__}: {str(e)[:90]}")

    outdir = PROJECT / "Analysis1" / "TP"
    outdir.mkdir(parents=True, exist_ok=True)
    for letter, (name, _) in zip(letters, MODELS):
        np.save(outdir / f"COM_{letter}.npy", np.asarray(grid[name], dtype=float))
    log(f"saved COM_A..COM_H.npy to {outdir.relative_to(PROJECT)}")

    metric_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    blocks = []
    for mi, metric in enumerate(metric_names):
        hdr = f"{metric:<14}" + "".join(f"{int(p*100):>10}%" for p in pcts)
        rows = [hdr, "-" * len(hdr)]
        for name, _ in MODELS:
            rows.append(f"{name:<14}" +
                        "".join(f"{grid[name][i][mi]:>11.4f}" for i in range(len(pcts))))
        blocks.append("\n".join(rows))
    table = "\n\n".join(blocks)
    print("\n" + table + "\n")

    dst = OUT / f"evaluation_sweep_ep{args.epochs}.txt"
    dst.write_text(
        f"Paper 1 (SMA-CLMPNet) - comparative analysis sweep over training percentage\n"
        f"features : Features/Features.pkl (pre-extracted FaceForensics++)\n"
        f"samples  : {len(lab)} videos, class balance "
        f"{dict(zip(*[x.tolist() for x in np.unique(lab, return_counts=True)]))}\n"
        f"epochs   : {args.epochs}   (paper uses 500)\n"
        f"skipped  : {sorted(skip) or 'none'}\n"
        f"elapsed  : {time.time()-t_all:.0f}s\n"
        f"columns  : training percentage 40..90%\n\n" + table + "\n", encoding="utf-8")
    log(f"wrote {dst.relative_to(PROJECT)}  (total {time.time()-t_all:.0f}s)")
    return 0


def _kfold(args, data, lab):
    """Replicate KFAnalysis.ComparativeAnalysis: KFold(n_splits=k, shuffle=True,
    random_state=1) for k in 6..10, metrics averaged across the evaluated folds,
    saved to Analysis1/KF/COM_<A..H>.npy.

    Two deviations, both forced and both documented in SKILL.md:
      * KFAnalysis.train_test_split loops `range(len(data['image']))`, but
        GetData never stores an 'image' key - the real method raises KeyError on
        the shipped pickle. We index on len(data['labels']) instead, which is the
        same length and clearly the intent.
      * evaluating every fold of every k is 6+7+8+9+10 = 40 fold-fits x 7 models
        (~9.5 h here), so --folds-per-k caps how many folds of each k are run.
    """
    import numpy as np
    from sklearn.model_selection import KFold
    from SubFunctions.Model import Network
    from SubFunctions.Evaluate import Evaluation_Metrics

    skip = set((args.skip or "").split(",")) - {""}
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    ks = [6, 7, 8, 9, 10]
    n = len(np.asarray(data["labels"]))
    grid = {name: [] for name, _ in MODELS}
    t_all = time.time()

    for k in ks:
        kf = KFold(n_splits=k, random_state=1, shuffle=True)
        per_model = {name: [] for name, _ in MODELS}
        nfolds = 0
        for j, (tr, te) in enumerate(kf.split(np.arange(n))):
            if j >= args.folds_per_k:
                break
            nfolds += 1
            sel = lambda key, idx: np.asarray([data[key][i] for i in idx])
            keys = ["comparative1", "comparative2", "comparative3",
                    "comparative4", "comparative5", "proposed"]
            params = {f"x_train{i+1}": sel(kk, tr) for i, kk in enumerate(keys)}
            params.update({f"x_test{i+1}": sel(kk, te) for i, kk in enumerate(keys)})
            y_tr = np.asarray([data["labels"][i] for i in tr]).astype(int)
            y_te = np.asarray([data["labels"][i] for i in te]).astype(int)
            params.update({"y_train": y_tr, "y_test": y_te, "epochs": args.epochs})
            log(f"===== k={k} fold {j+1}/{min(k, args.folds_per_k)}  "
                f"(train n={len(tr)}, test n={len(te)}) =====")
            for name, fn in MODELS:
                if name in skip:
                    per_model[name].append([np.nan] * 5)
                    continue
                t = time.time()
                try:
                    net = Network(**params)
                    if args.batch_size:
                        net.batch_size = args.batch_size
                    m = Evaluation_Metrics(y_te, np.asarray(fn(net)))
                    per_model[name].append(list(m))
                    log(f"  {name:<14} acc={m[0]:.4f} f1={m[4]:.4f} ({time.time()-t:.1f}s)")
                except Exception as e:
                    per_model[name].append([np.nan] * 5)
                    log(f"  {name:<14} FAILED {type(e).__name__}: {str(e)[:80]}")
        for name, _ in MODELS:
            grid[name].append(np.nanmean(np.asarray(per_model[name], float), axis=0)
                              if nfolds else [np.nan] * 5)
        log(f"  -> k={k} averaged over {nfolds} fold(s)")

    outdir = PROJECT / "Analysis1" / "KF"
    outdir.mkdir(parents=True, exist_ok=True)
    for letter, (name, _) in zip(letters, MODELS):
        np.save(outdir / f"COM_{letter}.npy", np.asarray(grid[name], dtype=float))
    log(f"saved COM_A..COM_H.npy to {outdir.relative_to(PROJECT)}")

    metric_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    blocks = []
    for mi, metric in enumerate(metric_names):
        hdr = f"{metric:<14}" + "".join(f"{'k='+str(k):>11}" for k in ks)
        rows = [hdr, "-" * len(hdr)]
        for name, _ in MODELS:
            rows.append(f"{name:<14}" +
                        "".join(f"{grid[name][i][mi]:>11.4f}" for i in range(len(ks))))
        blocks.append("\n".join(rows))
    table = "\n\n".join(blocks)
    print("\n" + table + "\n")
    dst = OUT / f"evaluation_kfold_ep{args.epochs}.txt"
    dst.write_text(
        f"Paper 1 (SMA-CLMPNet) - K-fold comparative analysis\n"
        f"features : Features/Features.pkl (pre-extracted FaceForensics++)\n"
        f"samples  : {n} videos\n"
        f"epochs   : {args.epochs}   (paper uses 500)\n"
        f"folds    : first {args.folds_per_k} fold(s) of each k, averaged\n"
        f"skipped  : {sorted(skip) or 'none'}\n"
        f"elapsed  : {time.time()-t_all:.0f}s\n\n" + table + "\n", encoding="utf-8")
    log(f"wrote {dst.relative_to(PROJECT)}  (total {time.time()-t_all:.0f}s)")
    return 0


def _epoch_curve(args, data, lab):
    """Stand-in for TPAnalysis.PerformanceAnalysis: the proposed model only,
    across an increasing epoch budget, so the trend with training length is
    visible. The paper sweeps 100..500; we sweep whatever --curve gives."""
    import numpy as np
    from SubFunctions.Analysis import train_test_split
    from SubFunctions.Model import Network
    from SubFunctions.Evaluate import Evaluation_Metrics

    budgets = [int(x) for x in args.curve.split(",")]
    d = train_test_split(data, train_size=args.train_pct)
    base = {f"x_train{i+1}": d[i] for i in range(6)}
    base.update({f"x_test{i+1}": d[6 + i] for i in range(6)})
    y_te = np.asarray(d[13])
    rows, t_all = [], time.time()
    log(f"epoch curve on SMA-CLMPNet, train_size={args.train_pct}, "
        f"test n={len(y_te)}, budgets={budgets}")
    for ep in budgets:
        p = dict(base, y_train=d[12], y_test=d[13], epochs=ep)
        t = time.time()
        net = Network(**p)
        m = Evaluation_Metrics(y_te, np.asarray(net.ThreeDCNNLSTM(opt=3)))
        rows.append((ep, *m, time.time() - t))
        log(f"  epochs={ep:<4} acc={m[0]:.4f} sens={m[1]:.4f} spec={m[2]:.4f} "
            f"prec={m[3]:.4f} f1={m[4]:.4f} ({time.time()-t:.1f}s)")
    hdr = f"{'Epochs':<8}{'Accuracy':>10}{'Sensitivity':>13}{'Specificity':>13}{'Precision':>11}{'F1':>9}{'sec':>9}"
    lines = [hdr, "-" * len(hdr)] + [
        f"{e:<8}{a:>10.4f}{se:>13.4f}{sp:>13.4f}{pr:>11.4f}{f:>9.4f}{s:>9.1f}"
        for e, a, se, sp, pr, f, s in rows]
    table = "\n".join(lines)
    print("\n" + table + "\n")
    dst = OUT / "evaluation_epoch_curve.txt"
    dst.write_text(
        f"Paper 1 - SMA-CLMPNet performance vs training epochs\n"
        f"train_size={args.train_pct}, test n={len(y_te)}, "
        f"elapsed {time.time()-t_all:.0f}s\n\n" + table + "\n", encoding="utf-8")
    log(f"wrote {dst.relative_to(PROJECT)}")
    return 0


def cmd_evaluate(args):
    """Train + evaluate on the FaceForensics++ features that ship in
    Features/Features.pkl - i.e. Main.py's 'Yes' branch WITHOUT needing the raw
    videos, which are only required to re-extract those features.

    This mirrors one iteration of TPAnalysis.ComparativeAnalysis: same
    train_test_split, same Network models, same Evaluation_Metrics. Epoch count
    and training percentage are exposed so it can be run at a sane scale; the
    paper's numbers come from epochs=500 over 6 training percentages.
    """
    import numpy as np
    from SubFunctions.GetData import ReadDataset
    from SubFunctions.Analysis import train_test_split
    from SubFunctions.Model import Network
    from SubFunctions.Evaluate import Evaluation_Metrics

    t0 = time.time()
    data = ReadDataset(exec=False).read_data()
    log(f"features loaded in {time.time() - t0:.1f}s")
    for k, v in data.items():
        v = np.asarray(v)
        log(f"    {k:<14} {str(v.shape):<24} {v.dtype}")
    lab = np.asarray(data["labels"])
    u, c = np.unique(lab, return_counts=True)
    log(f"    class balance: {dict(zip(u.tolist(), c.tolist()))}  (0=original, 1=manipulated)")

    if args.sweep:
        return _sweep(args, data, lab)
    if args.kfold:
        return _kfold(args, data, lab)
    if args.curve:
        return _epoch_curve(args, data, lab)

    tp = args.train_pct
    log(f"train_test_split at train_size={tp}, epochs={args.epochs}")
    d = train_test_split(data, train_size=tp)
    params = {f"x_train{i+1}": d[i] for i in range(6)}
    params.update({f"x_test{i+1}": d[6 + i] for i in range(6)})
    params.update({"y_train": d[12], "y_test": d[13], "epochs": args.epochs})
    y_test = np.asarray(d[13])
    log(f"    train n={len(np.asarray(d[12]))}  test n={len(y_test)}")

    wanted = args.models.split(",") if args.models else [m[0] for m in MODELS]
    rows, failures = [], []
    for name, fn in MODELS:
        if name not in wanted:
            continue
        log(f"--- {name} ---")
        t = time.time()
        try:
            net = Network(**params)
            if args.batch_size:
                net.batch_size = args.batch_size
            # BA-TFD (ViTDCNN) runs MultiHeadAttention over an undownsampled
            # 128x128 map - its MaxPooling2D(1,1) layers are no-ops - so the
            # attention tensor is enormous and it OOMs at the default
            # batch_size=32. Only the batch size is lowered; the architecture
            # is left exactly as published.
            if name == "BA-TFD" and not args.batch_size:
                net.batch_size = 2
                log("    batch_size lowered to 2 (see ViTDCNN OOM note)")
            pred = fn(net)
            acc, sen, spe, pre, f1 = Evaluation_Metrics(y_test, np.asarray(pred))
            dt = time.time() - t
            rows.append((name, acc, sen, spe, pre, f1, dt))
            log(f"    acc={acc:.4f} sens={sen:.4f} spec={spe:.4f} "
                f"prec={pre:.4f} f1={f1:.4f}  ({dt:.1f}s)")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            log(f"    FAILED {type(e).__name__}: {e}")
            traceback.print_exc()

    if rows:
        hdr = f"{'Model':<14}{'Accuracy':>10}{'Sensitivity':>13}{'Specificity':>13}{'Precision':>11}{'F1':>9}{'sec':>9}"
        lines = [hdr, "-" * len(hdr)]
        for n, a, se, sp, p, f, dt in rows:
            lines.append(f"{n:<14}{a:>10.4f}{se:>13.4f}{sp:>13.4f}{p:>11.4f}{f:>9.4f}{dt:>9.1f}")
        table = "\n".join(lines)
        print("\n" + table + "\n")
        dst = OUT / f"evaluation_tp{int(tp*100)}_ep{args.epochs}.txt"
        dst.write_text(
            f"Paper 1 (SMA-CLMPNet) - reduced evaluation\n"
            f"features : Features/Features.pkl (pre-extracted FaceForensics++)\n"
            f"samples  : {len(lab)} videos, class balance {dict(zip(u.tolist(), c.tolist()))}\n"
            f"split    : train_size={tp}  (train n={len(np.asarray(d[12]))}, test n={len(y_test)})\n"
            f"epochs   : {args.epochs}   (paper uses 500 over 6 training percentages)\n\n"
            + table + "\n", encoding="utf-8")
        log(f"wrote {dst.relative_to(PROJECT)}")

    if failures:
        log("FAILED MODELS: " + ", ".join(f"{k} ({m})" for k, m in failures))
        return 1
    log("EVALUATION OK")
    return 0


# --------------------------------------------------------------------------
# gui
# --------------------------------------------------------------------------
STAGES = [
    ("select", "select_data_event", "Select Video"),
    ("preprocess", "preprocessing_event", "Preprocessing"),
    ("gradcam", "get_gradcam", "GradCAM"),
    ("resnet", "get_resnetstat", "Resnet Statistical"),
    ("vgg", "get_vgg", "VGG LDZP"),
    ("flow", "get_flow", "Optical Flow"),
]


class Shooter:
    """Screengrab of the Tk window.

    Getting the capture rectangle right on Windows is fiddly: Tk's
    winfo_rootx/rooty/width/height are in Tk's own coordinate space, while
    PIL.ImageGrab works in physical screen pixels. On a scaled display they
    disagree, and customtkinter adds a *second* scaling factor of its own on
    top - so deriving a ratio from the screen size overshoots and you capture
    the desktop around the window (verified: got the taskbar).

    Win32 GetWindowRect on the toplevel HWND sidesteps all of it and returns
    exactly the rectangle ImageGrab needs - provided the process is DPI-aware,
    which setup_dpi() guarantees.
    """

    def __init__(self, root):
        from PIL import ImageGrab
        self.grab = ImageGrab.grab
        self.root = root
        self.n = 0
        # winfo_id() is the Tk child window; walk up to the real toplevel.
        self.hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)  # GA_ROOT
        log(f"toplevel hwnd {self.hwnd}, rect {self.rect()}")

    def rect(self):
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        r = RECT()
        if self.hwnd and ctypes.windll.user32.GetWindowRect(self.hwnd, ctypes.byref(r)):
            return (r.left, r.top, r.right, r.bottom)
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        return (x, y, x + self.root.winfo_width(), y + self.root.winfo_height())

    def pump(self, seconds=0.4):
        """Tk only repaints inside mainloop(); we never call mainloop(), so
        the window stays blank unless events are pumped by hand."""
        end = time.time() + seconds
        while time.time() < end:
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.02)

    def shot(self, name):
        self.n += 1
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.pump(0.6)
        img = self.grab(bbox=self.rect(), all_screens=True)
        dst = SHOTS / f"{self.n:02d}-{name}.png"
        img.save(dst)
        log(f"  screenshot {dst.relative_to(PROJECT)}  {img.size}")
        return dst


def setup_dpi():
    """Must run BEFORE tkinter/customtkinter is imported. Without it the
    process is DPI-virtualised, Win32 window rects come back in logical units,
    and screenshots capture the wrong region on a scaled display."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor v2
        log("process DPI awareness set (per-monitor)")
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
        log("process DPI awareness set (system)")


def cmd_gui(args):
    setup_dpi()
    video = Path(args.video) if args.video else OUT / "sample.mp4"
    if not video.exists():
        log(f"no video at {video}; run 'make-video' first")
        return 1

    wanted = args.stages.split(",") if args.stages else [s[0] for s in STAGES]

    # Patch the blocking native file picker before GUI.py can call it.
    import tkinter.filedialog as fd
    fd.askopenfilename = lambda *a, **k: str(video)

    if not getattr(args, "full_package", False):
        subfunctions_lite()

    log("importing GUI (builds ResNet101 + VGG16 at module scope, ~30-90s)...")
    t0 = time.time()
    import GUI
    log(f"GUI imported in {time.time() - t0:.0f}s")

    app = GUI.App()
    GUI.app = app  # GUI.exit_event() is a staticmethod referencing global `app`
    sh = Shooter(app)
    sh.pump(0.8)
    sh.shot("launched")

    failures = []
    for key, method, label in STAGES:
        if key not in wanted:
            continue
        log(f"stage '{key}' -> App.{method}()  [{label}]")
        t = time.time()
        try:
            getattr(app, method)()
            sh.pump(0.5)
            sh.shot(key)
            log(f"  ok in {time.time() - t:.1f}s")
        except Exception as e:
            failures.append((key, f"{type(e).__name__}: {e}"))
            log(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                sh.shot(f"{key}-FAILED")
            except Exception:
                pass

    app.destroy()
    log(f"screenshots in {SHOTS.relative_to(PROJECT)}")
    if failures:
        log("FAILED STAGES: " + ", ".join(f"{k} ({m})" for k, m in failures))
        return 1
    log("GUI WALKTHROUGH OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    full = dict(action="store_true", dest="full_package",
                help="run SubFunctions/__init__.py for real (needs a working torch)")
    sub.add_parser("check")
    sub.add_parser("make-video")
    p = sub.add_parser("plots")
    p.add_argument("--full-package", **full)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--epochs", type=int, default=5,
                    help="training epochs per model (paper uses 500)")
    ev.add_argument("--train-pct", type=float, default=0.9, dest="train_pct",
                    help="training fraction (paper sweeps 0.4-0.9)")
    ev.add_argument("--models", help="comma list: " + ",".join(m[0] for m in MODELS))
    ev.add_argument("--batch-size", type=int, dest="batch_size",
                    help="override Network.batch_size (default 32; BA-TFD forced to 2)")
    ev.add_argument("--sweep", action="store_true",
                    help="run the full 40-90%% training-percentage comparative sweep")
    ev.add_argument("--skip", help="comma list of models to skip in --sweep")
    ev.add_argument("--kfold", action="store_true",
                    help="K-fold comparative analysis for k=6..10")
    ev.add_argument("--folds-per-k", type=int, default=2, dest="folds_per_k",
                    help="how many folds of each k to evaluate (default 2)")
    ev.add_argument("--curve",
                    help="comma list of epoch budgets, e.g. 2,5,10,20,40")
    g = sub.add_parser("gui")
    g.add_argument("--video", help="path to an .mp4/.avi to feed Select Video")
    g.add_argument("--stages", help="comma list: " + ",".join(s[0] for s in STAGES))
    g.add_argument("--full-package", **full)
    a = sub.add_parser("all")
    a.add_argument("--video")
    a.add_argument("--stages")
    a.add_argument("--full-package", **full)
    args = ap.parse_args()

    setup()
    if args.cmd == "all":
        for fn in (cmd_check, cmd_make_video, cmd_plots):
            rc = fn(args)
            if rc:
                return rc
        return cmd_gui(args)
    return {"check": cmd_check, "make-video": cmd_make_video,
            "plots": cmd_plots, "gui": cmd_gui,
            "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
