"""Rewrite the results paragraphs of the Paper 1 .docx with measured numbers.

The paragraphs currently in the document - both the authors' originals and the
reproduction text written on 2026-08-04/05 - report scores that came from the
tampered metric in mealpy/metrics.py and are not measurements. This script
replaces them with numbers scored by Optimized/metrics_fixed.py.

Full text of each anchored paragraph is replaced, keeping that paragraph's own
formatting: the new text goes into its first run and the remaining runs are
emptied. A timestamped backup is written next to the document.
"""
import datetime
import html
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARA = re.compile(r"<w:p[ >].*?</w:p>", re.S)
RUN = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)

P = Path(r"C:\Users\USER\Downloads\PostDoc\Implimentation_Paper1")
DOC = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    P / "Neha Dhiman-Paper 1 ---final.docx"

PUBLISHED = ["EfficientNet", "STIDNet", "DCNN", "GLCM",
             "MUSE-CLMPNet", "SCAM-CLMPNet", "SMA-CLMPNet"]
LATEST = ["EfficientNetV2S", "ConvNeXtTiny", "MobileNetV3Large", "ResNetRS50"]

INTEGRITY = (
    "These figures replace the values previously reported in this section. "
    "Those values were produced by SubFunctions/Evaluate.py, which computes "
    "every metric from mealpy.metrics.confusion_matrix; the vendored copy of "
    "that library has a modified _check_targets that discards the "
    "classifier's predictions and substitutes the ground-truth labels with a "
    "uniformly random fraction of entries flipped, so the numbers it returns "
    "are independent of the model, the features and the training. A perfect "
    "predictor scores between 64.5% and 100.0% across repeated calls on that "
    "path, and an inverted predictor scores no worse. The figures given here "
    "were computed instead from a real confusion matrix, using the same "
    "metric definitions as the original code."
)

CORPUS = (
    "The evaluation corpus is 50 videos, 29 authentic and 21 forged, so the "
    "test partition ranges from 31 videos at the 40% split to 6 at the 90% "
    "split and a single misclassification moves accuracy by between 3.23 and "
    "16.67 percentage points. Balanced accuracy is reported alongside "
    "accuracy because a model that predicts one class for every input scores "
    "58.00% accuracy on this corpus while learning nothing; such a model "
    "scores 50.00% balanced accuracy. BA-TFD is omitted throughout: its "
    "ViTDCNN definition applies MaxPooling2D(1, 1), which performs no "
    "downsampling, so the flattened 1,048,576-element vector entering "
    "Dense(2048) requires an 8.6 GB weight matrix and exhausts memory at "
    "every batch size tested."
)


def load(dirname):
    d = P / "Analysis1" / dirname
    man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    return {f.stem: np.load(f) for f in d.glob("*.npy")}, man


def plain(p):
    return html.unescape(re.sub(r"<[^>]+>", "", p))


def replace_para(doc, anchor, new_text):
    hits = [m for m in PARA.finditer(doc) if anchor in plain(m.group(0))]
    if len(hits) != 1:
        raise SystemExit(f"anchor matched {len(hits)} paragraphs: {anchor[:70]!r}")
    m = hits[0]
    para = m.group(0)
    spans = [(x.start(1), x.end(1)) for x in RUN.finditer(para)]
    if not spans:
        raise SystemExit("anchor paragraph has no text runs")
    out = para
    for i, (s, e) in enumerate(reversed(spans)):
        idx = len(spans) - 1 - i
        body = html.escape(new_text, quote=False) if idx == 0 else ""
        out = out[:s] + body + out[e:]
    return out.replace("<w:t>", '<w:t xml:space="preserve">', 1), m


def pct(v):
    return "not available" if v is None or np.isnan(v) else f"{v*100:.2f}%"


def listing(res, names, col, row):
    parts = [f"{n} {pct(res[n][row, col])}" for n in names
             if n in res and not np.isnan(res[n][row, col])]
    return ", ".join(parts)


def mean_listing(res, names, col):
    parts = [f"{n} {pct(np.nanmean(res[n][:, col]))}" for n in names
             if n in res and not np.all(np.isnan(res[n][:, col]))]
    return ", ".join(parts)


def degenerate(res, names):
    """Models that emitted a CONSTANT prediction, counted per split.

    Judging this from mean sensitivity/specificity is wrong: a model that is
    constant on some splits and not others averages to something that looks
    discriminating. A constant prediction on a split shows up as sensitivity
    or specificity being exactly zero on that split.
    """
    out = []
    for n in names:
        if n not in res:
            continue
        a = res[n]
        k = int(np.sum(np.minimum(np.nan_to_num(a[:, 1]),
                                  np.nan_to_num(a[:, 2])) == 0.0))
        if k:
            out.append((n, k, a.shape[0]))
    return out


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def build_561(res, man):
    pcts = [int(round(p * 100)) for p in man["train_pcts"]]
    r = len(pcts) - 1
    names = [n for n in PUBLISHED if n in res]
    others = [n for n in names if n != "SMA-CLMPNet"]
    dead = degenerate(res, names)
    span = (f"from {pcts[0]}% to {pcts[-1]}%" if len(pcts) > 1
            else f"at {pcts[0]}%")
    t = (
        "The comparison of the SMA-CLMPNet with existing approaches by varying "
        f"training percentages {span} using the Face "
        "Forensics++ dataset is demonstrated in Figure 9. At the "
        f"{pcts[r]}% training percentage the SMA-CLMPNet attained an accuracy "
        f"of {pct(res['SMA-CLMPNet'][r, 0])} with a balanced accuracy of "
        f"{pct(res['SMA-CLMPNet'][r, 5])}, whereas the established methods "
        f"attained: {listing(res, others, 0, r)}. Averaged over the "
        f"{WORDS.get(len(pcts), len(pcts))} evaluated training "
        f"{'percentage' if len(pcts) == 1 else 'percentages'} the accuracies were: "
        f"{mean_listing(res, names, 0)}; the corresponding balanced "
        f"accuracies were {mean_listing(res, names, 5)}. Mean sensitivities "
        f"were {mean_listing(res, names, 1)} and mean specificities "
        f"{mean_listing(res, names, 2)}, with mean F1-scores of "
        f"{mean_listing(res, names, 4)}. "
    )
    if dead:
        listing_txt = ", ".join(f"{n} on {k} of {tot}" for n, k, tot in dead)
        t += (
            "The sensitivity and specificity columns expose a failure that a "
            "headline accuracy figure conceals: on a number of splits these "
            "methods assigned a single label to every test video, giving a "
            "sensitivity or specificity of exactly zero and a balanced "
            f"accuracy of 50%. The affected counts are {listing_txt} "
            "evaluated splits. With 19 to 44 training samples and a batch "
            "size of 32, training for 10 epochs performs at most two gradient "
            "updates per epoch, which is not sufficient for these networks to "
            "depart from the majority class. "
        )
    lat = [n for n in LATEST if n in res]
    if lat:
        t += (
            "Current-generation backbones evaluated on the same splits and the "
            "same features - each as a frozen ImageNet feature extractor with "
            "a trained classifier head, since a training split of 19 to 44 "
            f"samples cannot fine-tune them - reached: "
            f"{mean_listing(res, lat, 0)} mean accuracy "
            f"({mean_listing(res, lat, 5)} balanced). They do not separate the "
            "classes either, which indicates that the limitation lies in the "
            "extracted features and the size of the corpus rather than in any "
            "one architecture. "
        )
    if "SMA-CLMPNet-Opt" in res:
        o = res["SMA-CLMPNet-Opt"]
        b = res["SMA-CLMPNet"]
        t += (
            "Retraining the SMA-CLMPNet with an optimised recipe - the "
            "architecture unchanged, but the batch size reduced from 32 to 8 "
            "so that each epoch performs several gradient updates rather than "
            "two, the inputs standardised using training-split statistics "
            "only, class weights applied for the 29/21 imbalance, and the "
            "learning rate cosine-decayed over a longer budget - gave a mean "
            f"accuracy of {pct(np.nanmean(o[:, 0]))} and a mean balanced "
            f"accuracy of {pct(np.nanmean(o[:, 5]))}, against "
            f"{pct(np.nanmean(b[:, 0]))} and {pct(np.nanmean(b[:, 5]))} for "
            "the published recipe. The recipe therefore does not rescue the "
            "model: it changes which class the network collapses onto without "
            "producing a classifier that separates them. "
        )
    return t + INTEGRITY + " " + CORPUS


def build_562(res, man):
    ks = man["k_values"]
    r = len(ks) - 1
    names = [n for n in PUBLISHED if n in res]
    others = [n for n in names if n != "SMA-CLMPNet"]
    t = (
        "The SMA-CLMPNet is compared with existing methods by different k-fold "
        f"values of {', '.join(str(k) for k in ks)} using the Face "
        "Forensics++ dataset, and the graphical representation is depicted in "
        f"Figure 10. At k-fold {ks[r]} the SMA-CLMPNet achieved "
        f"{pct(res['SMA-CLMPNet'][r, 0])} accuracy and "
        f"{pct(res['SMA-CLMPNet'][r, 5])} balanced accuracy, compared against "
        f"{listing(res, others, 0, r)}. Averaged over k = "
        f"{ks[0]} to {ks[-1]} the accuracies were "
        f"{mean_listing(res, names, 0)} and the balanced accuracies "
        f"{mean_listing(res, names, 5)}. Mean sensitivities were "
        f"{mean_listing(res, names, 1)} and mean specificities "
        f"{mean_listing(res, names, 2)}. "
        f"Folds were generated with stratified k-fold partitioning, "
        f"{man['folds_per_k']} fold evaluated per k value. The published "
        "KFAnalysis routine could not be used: it indexes data['image'], a key "
        "that ReadDataset never stores, and it scores through the same "
        "compromised metric. "
    )
    return t + INTEGRITY + " " + CORPUS


def build_58(res, man):
    names = ([n for n in PUBLISHED if n in res]
             + [n for n in LATEST if n in res]
             + [n for n in ["SMA-CLMPNet-Opt"] if n in res])
    stat = "; ".join(
        f"{n}: best {pct(np.nanmax(res[n][:, 0]))}, mean "
        f"{pct(np.nanmean(res[n][:, 0]))}, variance "
        f"{np.nanvar(res[n][:, 0]):.6f}"
        for n in names if not np.all(np.isnan(res[n][:, 0])))
    return (
        "The statistical analysis of the SMA-CLMPNet with the existing "
        "approaches using the Face Forensics++ dataset is depicted in Table 2. "
        "The analysis showcases the best, mean, and variance values of the "
        "accuracy attained by the methods across the evaluated training "
        f"percentages: {stat}. The variances are dominated by the size of the "
        "evaluation corpus rather than by any property of the methods. "
        + INTEGRITY + " " + CORPUS)


def main():
    res, man = load("TRUE")
    edits = [("The comparison of the SMA-CLMPNet with existing approaches by "
              "varying training percentages", build_561(res, man)),
             ("The statistical analysis of the SMA-CLMPNet with the existing "
              "approaches", build_58(res, man))]

    kf_dir = P / "Analysis1" / "TRUE_KF"
    if (kf_dir / "run_manifest.json").exists():
        kres, kman = load("TRUE_KF")
        edits.insert(1, ("The SMA-CLMPNet is compared with existing methods by "
                         "different k-fold values", build_562(kres, kman)))
    else:
        print("!! 5.6.2 left untouched - no measured k-fold results yet")

    with zipfile.ZipFile(DOC) as z:
        items = [(i, z.read(i.filename)) for i in z.infolist()]
    doc = next(b for i, b in items
               if i.filename == "word/document.xml").decode("utf8")

    for anchor, text in edits:
        new_para, m = replace_para(doc, anchor, text)
        doc = doc[:m.start()] + new_para + doc[m.end():]
        print(f"rewrote {anchor[:52]!r} -> {len(text)} chars")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(DOC, f"{DOC}.bak-{stamp}")
    tmp = str(DOC) + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info, dat in items:
            if info.filename == "word/document.xml":
                dat = doc.encode("utf8")
            z.writestr(info, dat)
    os.replace(tmp, DOC)
    print(f"backup: {os.path.basename(DOC)}.bak-{stamp}")
    print("done")


if __name__ == "__main__":
    main()
