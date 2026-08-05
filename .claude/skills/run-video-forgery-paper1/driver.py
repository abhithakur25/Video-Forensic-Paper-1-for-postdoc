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
            "plots": cmd_plots, "gui": cmd_gui}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
