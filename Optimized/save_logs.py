"""Tidy and index the run logs.

Collapses repeated noise (sklearn warnings, Keras progress bars, TF device
messages) into counted placeholders, keeps every informative line, prefixes
each file with a description, and regenerates logs/README.md.

Idempotent: re-running does not re-collapse an already-collapsed file, and a
log still being written is indexed but left untouched.
"""
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
LOGS = P / "logs"
MARK = "# [tidied]"

# filename -> description
DESC = {
    "sweep_true.log":
        "The corrected re-evaluation. All 7 of the paper's models plus the 4 "
        "current-generation backbones and SMA-CLMPNet-Opt, across training "
        "percentages 40-90%, scored with a real confusion matrix. 10,109 s. "
        "Source of Analysis1/TRUE and of the tables in sections 5.6.1 / 5.8.",
    "kfold_true.log":
        "K-fold comparison, k = 6..10, stratified folds, correct scoring. "
        "Feeds section 5.6.2. The published KFAnalysis could not be used: "
        "Analysis.py:355 indexes data['image'], a key ReadDataset never "
        "stores.",
    "kfold_true_interrupted.log":
        "First k-fold attempt. Died when the session that launched it exited, "
        "4 of 12 models into k=6 and before the first checkpoint.",
    "kfold_true_interrupted2.log":
        "Second k-fold attempt. Died 6 models into k=6 when the agent task "
        "that had called Start-Process was reaped - Start-Process does not "
        "escape the harness job object. Fixed by launching through WMI "
        "(Optimized/run_kfold.bat) and adding --resume to kfold mode.",
    "optimize_v2.log":
        "Representation and model search: 14 representations x 9 model "
        "families under nested cross-validation, with a 100-shuffle "
        "permutation test on the winner. Established the temporal-delta "
        "signal at p = 0.0099.",
    "optimize_v2_frames.log":
        "Second search pass over the per-frame backbone embeddings, which the "
        "first pass missed because they were written after it began.",
    "optimize_v3.log":
        "Higher-order temporal features (acceleration, lag-2, "
        "autocorrelation) and stacked/voting ensembles. Every addition scored "
        "BELOW plain L1 logistic regression on first-order deltas.",
    "optimize_weights.log":
        "Class-weight, probability-calibration and decision-threshold sweep, "
        "30 configurations, all selected inside training folds. Best 69.67%, "
        "below the untuned 77.17%.",
    "stil_tim.log":
        "STIL's Temporal Inconsistency Module (Gu et al., ACM MM 2021) on "
        "this feature tensor. TIM_Module and ISM_Module imported unmodified "
        "from Tencent/TFace 171ec143 - the two repos everyone cites for STIL "
        "contain no code and both redirect there. 26,696 parameters against "
        "SMA-CLMPNet's 2,258,534. Pooled out-of-fold balanced accuracy "
        "50.49%, catching 6 of 21 forgeries; three of five folds checkpointed "
        "at epoch 1 or 3, so nothing after initialisation improved "
        "validation.",
    "frame_embeddings.log":
        "Per-frame and frame-difference backbone embeddings of the 'proposed' "
        "tensor (MobileNetV3Large 15,360-dim, EfficientNetV2S 20,480-dim).",
    "paper2_model_500.log":
        "Paper 2's BiLSTMGBM ported to Paper 1's features at its own settings "
        "(500 epochs, batch 32, incremental learning), omitting the test-set "
        "weight fitting. Lands at chance.",
    "final_tables.log":
        "Full metric tables by training percentage for the best honest "
        "pipeline.",
    "keras_weight_download.log":
        "First import of SubFunctions: ResNet101 (180 MB) and VGG16 (553 MB) "
        "download at module scope - the 733 MB cost of a bare import.",
    "conda_pytorch_ssl_failure.log":
        "conda install pytorch FAILED with CERTIFICATE_VERIFY_FAILED through "
        "the network's TLS interception; why torch stayed unavailable and the "
        "SubFunctions __init__ bypass was needed.",
    "webapp_start_failure.log":
        "First webapp launch: AttributeError, module 'cv2' has no attribute "
        "'data'. Fixed by bundling the Haar cascade.",
    "webapp_server.log":
        "Flask dev server serving the analyze API during verification.",
    "git_push_initial.log": "Initial push of 1,672 files to GitHub.",
    "session_notes.md": "Running notes: every error hit and the fix.",
}

# (regex, label) - consecutive matching lines collapse to one counted line
NOISE = [
    (re.compile(r"site-packages\\sklearn\\feature_selection"), "sklearn feature-selection warning"),
    (re.compile(r'warnings\.warn\("Features %s are constant'), "constant-feature warning"),
    (re.compile(r"^\s*f = msb / msw\s*$"), "divide-by-zero in f_classif"),
    (re.compile(r"invalid value encountered in (true_divide|divide)"), "numpy invalid-value warning"),
    (re.compile(r"\[=*[.>]+=*\].*ETA:"), "Keras progress bar"),
    (re.compile(r"(cudart64|cublas|cudnn|cufft|curand|cusolver|cusparse).*dlerror"), "CUDA library probe"),
    (re.compile(r"Could not load dynamic library|Ignore above cudart dlerror"), "TF device message"),
    (re.compile(r"This TensorFlow binary is optimized|To enable them in other operations"), "TF CPU-feature notice"),
    (re.compile(r"tf\.function retracing|reduce_retracing"), "TF retracing warning"),
    (re.compile(r"^\s*_warn\(\(\"h5py|h5py.*HDF5"), "h5py version warning"),
]


def classify(line):
    for rx, label in NOISE:
        if rx.search(line):
            return label
    return None


def tidy(path):
    raw = path.read_text(encoding="utf-8", errors="replace").split("\n")
    if raw and raw[0].startswith(MARK):
        return None
    out, i, dropped = [], 0, 0
    while i < len(raw):
        lab = classify(raw[i])
        if lab is None:
            out.append(raw[i])
            i += 1
            continue
        j = i
        while j < len(raw) and classify(raw[j]) is not None:
            j += 1
        n = j - i
        dropped += n
        out.append(f"    ... [{n} lines collapsed: {lab}]")
        i = j
    return out, dropped


def main():
    running = set()
    now = time.time()
    for f in LOGS.glob("*.log"):
        if now - f.stat().st_mtime < 120:
            running.add(f.name)

    rows = []
    for f in sorted(LOGS.glob("*.log")) + sorted(LOGS.glob("*.md")):
        if f.name == "README.md":
            continue
        desc = DESC.get(f.name, "")
        if f.name in running:
            rows.append((f.name, len(f.read_text("utf-8", "replace").split("\n")),
                         0, desc + "  **(still being written)**"))
            print(f"{f.name:<30} in progress - left untouched")
            continue
        if f.suffix == ".md":
            rows.append((f.name, len(f.read_text("utf-8", "replace").split("\n")),
                         0, desc))
            continue
        r = tidy(f)
        if r is None:
            n = len(f.read_text("utf-8", "replace").split("\n"))
            rows.append((f.name, n, 0, desc))
            print(f"{f.name:<30} already tidied")
            continue
        out, dropped = r
        before = f.stat().st_size
        header = (f"{MARK} {f.name}\n# {desc}\n"
                  f"# tidied {time.strftime('%Y-%m-%d %H:%M:%S')}; "
                  f"{dropped} noise lines collapsed\n"
                  + "#" + "-" * 74)
        f.write_text(header + "\n" + "\n".join(out), encoding="utf-8")
        after = f.stat().st_size
        rows.append((f.name, len(out), dropped, desc))
        print(f"{f.name:<30} {before/1e6:7.2f} MB -> {after/1e6:5.2f} MB  "
              f"({dropped} lines collapsed)")

    idx = ["# Run logs\n",
           "Every log behind the results. Repetitive noise (sklearn warnings, "
           "Keras progress bars, TensorFlow device probes) is collapsed into "
           "counted placeholders; every informative line is kept.\n",
           "Every log here records a run scored by "
           "`../Optimized/metrics_fixed.py`, or a diagnostic. The three "
           "console records of runs scored by the tampered "
           "`mealpy/metrics.py` were removed on 2026-08-06 along with the "
           "rest of the fabricated material — see "
           "[`../Optimized/PROVENANCE.md`](../Optimized/PROVENANCE.md).\n",
           "| Log | Lines | Collapsed | What it records |", "|---|---|---|---|"]
    for name, lines, dropped, desc in rows:
        idx.append(f"| [`{name}`]({name}) | {lines} | "
                   f"{dropped if dropped else '—'} | {desc} |")
    idx += ["",
            "Result arrays are in `../Analysis1/TRUE` (training-percentage "
            "sweep) and `../Analysis1/TRUE_KF` (k-fold); search results are "
            "the JSON files in `../Optimized/`.",
            "", f"Regenerate this index with `python Optimized/save_logs.py`."]
    (LOGS / "README.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print(f"\nwrote logs/README.md ({len(rows)} entries)")


if __name__ == "__main__":
    main()
