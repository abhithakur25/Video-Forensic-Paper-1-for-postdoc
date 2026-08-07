"""Load every measured artefact the manuscript reports.

One place that knows where results live and what shape they are, so the prose
modules never touch a file path. Anything missing raises rather than silently
producing an empty table - a manuscript that quietly drops a section because a
run had not finished is exactly the failure mode this project exists to fix.
"""
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]

# Column order of every .npy in Analysis1/, fixed by
# Optimized/metrics_fixed.py: evaluation_metrics returns [ACC, SEN, SPE, PRE,
# F1] and the sweep appends balanced accuracy.
COLS = ["ACC", "SEN", "SPE", "PRE", "F1", "BAL"]
ACC, SEN, SPE, PRE, F1, BAL = range(6)

# Display order: published cohort first, then the modern backbones added here.
ORDER = ["SMA-CLMPNet", "MUSE-CLMPNet", "SCAM-CLMPNet", "SMA-CLMPNet-Opt",
         "DCNN", "EfficientNet", "STIDNet", "GLCM",
         "EfficientNetV2S", "ConvNeXtTiny", "MobileNetV3Large", "ResNetRS50"]

PRETTY = {"EfficientNetV2S": "EfficientNetV2-S",
          "ConvNeXtTiny": "ConvNeXt-Tiny",
          "MobileNetV3Large": "MobileNetV3-Large",
          "ResNetRS50": "ResNet-RS-50",
          "SMA-CLMPNet-Opt": "SMA-CLMPNet-Opt"}

GROUP = {
    "SMA-CLMPNet": "proposed", "MUSE-CLMPNet": "ablation",
    "SCAM-CLMPNet": "ablation", "SMA-CLMPNet-Opt": "proposed (tuned)",
    "DCNN": "published cohort", "EfficientNet": "published cohort",
    "STIDNet": "published cohort", "GLCM": "published cohort",
    "EfficientNetV2S": "modern backbone", "ConvNeXtTiny": "modern backbone",
    "MobileNetV3Large": "modern backbone", "ResNetRS50": "modern backbone",
}


def pretty(m):
    return PRETTY.get(m, m)


def _json(name, required=True):
    f = P / "Optimized" / name
    if not f.exists():
        if required:
            raise FileNotFoundError(f"missing result artefact: {f}")
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _arrays(sub):
    d = P / "Analysis1" / sub
    man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    arrs = {}
    for m in man["models"]:
        f = d / f"{m}.npy"
        if not f.exists():
            raise FileNotFoundError(f"missing {f}")
        arrs[m] = np.load(f) * 100.0
    return arrs, man


def load():
    sweep, sweep_man = _arrays("TRUE")
    kf, kf_man = _arrays("TRUE_KF")
    d = {
        "sweep": sweep, "sweep_man": sweep_man,
        "kf": kf, "kf_man": kf_man,
        "pcts": [int(round(x * 100)) for x in sweep_man["train_pcts"]],
        "ks": [int(k) for k in kf_man["k_values"]],
        "v2": _json("optimize_v2.json"),
        "v3": _json("optimize_v3.json"),
        "weights": _json("optimize_weights.json"),
        "roc": _json("roc_confusion.json"),
        "audit": _json("corpus_audit.json"),
        "stil": _json("oof_stil_tim.json"),
        "paper2": _json("paper2_model.json"),
        "probe": _json("feature_probe.json"),
        "frame_level": _json("frame_level_summary.json", required=False),
        "fad": _json("fad_followup.json", required=False),
    }
    # Consistency checks the manuscript's prose depends on.
    n = d["audit"]["n"]
    assert n == d["roc"]["corpus"]["n"] == 50, "corpus size changed"
    assert d["audit"]["authentic"] == 29 and d["audit"]["forged"] == 21
    for m, a in {**sweep, **kf}.items():
        assert a.shape[1] == 6, f"{m}: expected 6 metric columns, got {a.shape}"
    assert set(sweep) == set(kf) == set(ORDER), \
        f"model set changed: {sorted(set(sweep) ^ set(ORDER))}"
    return d


def mean_bal(arr):
    return float(np.nanmean(arr[:, BAL]))


def mean_acc(arr):
    return float(np.nanmean(arr[:, ACC]))


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{nd}f}"


def majority_accuracy(d):
    """Accuracy of the constant 'authentic' answer on the whole corpus."""
    return 100.0 * d["audit"]["authentic"] / d["audit"]["n"]


def degenerate(arr):
    """Rows where the model answered one class for every input.

    Specificity 0 with sensitivity 100 is 'always authentic'; the mirror image
    is 'always forged'. Both give balanced accuracy exactly 50.
    """
    out = 0
    for r in arr:
        if (r[SPE] == 0 and r[SEN] == 100) or (r[SEN] == 0 and r[SPE] == 100):
            out += 1
    return out
