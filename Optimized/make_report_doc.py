"""Generate a complete .docx record of the work: every step and every result.

Writes the OOXML directly (no python-docx in this environment): a minimal but
valid Word package with styled headings, body text, code blocks and tables.

Output: Paper1_Complete_Work_Report.docx in the project root.
"""
import datetime
import glob
import html
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
OUT = P / "Paper1_Complete_Work_Report.docx"

# --------------------------------------------------------------- OOXML parts
CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _style(sid, name, sz, bold, color, before=200, after=100, mono=False):
    font = "Consolas" if mono else "Calibri"
    # built outside the f-string: Python 3.8 forbids backslashes in f-string
    # expressions, and nesting quotes here is what would need them
    b_tag = "<w:b/>" if bold else ""
    c_tag = '<w:color w:val="%s"/>' % color if color else ""
    return (f'<w:style w:type="paragraph" w:styleId="{sid}">'
            f'<w:name w:val="{name}"/>'
            f'<w:pPr><w:spacing w:before="{before}" w:after="{after}"/></w:pPr>'
            f'<w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}"/>'
            f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
            f'{b_tag}{c_tag}</w:rPr></w:style>')


STYLES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
          + _style("Title", "Title", 56, True, "1F3864", 0, 240)
          + _style("Heading1", "heading 1", 32, True, "1F3864", 320, 120)
          + _style("Heading2", "heading 2", 26, True, "2E5496", 260, 100)
          + _style("Heading3", "heading 3", 23, True, "404040", 220, 80)
          + _style("Normal", "Normal", 21, False, None, 60, 60)
          + _style("Code", "Code", 17, False, "333333", 60, 60, mono=True)
          + _style("Caption", "Caption", 18, False, "666666", 20, 160)
          + '</w:styles>')


def esc(t):
    return html.escape(str(t), quote=False)


def para(text, style="Normal", bold_prefix=None):
    runs = ""
    if bold_prefix:
        runs += (f'<w:r><w:rPr><w:b/></w:rPr>'
                 f'<w:t xml:space="preserve">{esc(bold_prefix)}</w:t></w:r>')
    runs += f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{runs}</w:p>'


def bullet(text):
    return para("•  " + text, "Normal")


def code(lines):
    if isinstance(lines, str):
        lines = lines.split("\n")
    return "".join(para(l if l else " ", "Code") for l in lines)


def cell(text, bold=False, shade=None, width=1200):
    sh = f'<w:shd w:val="clear" w:fill="{shade}"/>' if shade else ""
    rpr = "<w:b/>" if bold else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{sh}</w:tcPr>'
            f'<w:p><w:pPr><w:pStyle w:val="Normal"/>'
            f'<w:spacing w:before="20" w:after="20"/></w:pPr>'
            f'<w:r><w:rPr>{rpr}<w:sz w:val="18"/></w:rPr>'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p></w:tc>')


def table(head, rows, widths=None, highlight=None):
    n = len(head)
    widths = widths or [max(1000, int(9000 / n))] * n
    borders = ('<w:tblBorders>' + "".join(
        f'<w:{s} w:val="single" w:sz="4" w:color="BFBFBF"/>'
        for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + '</w:tblBorders>')
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>{borders}</w:tblPr>']
    out.append("<w:tr>" + "".join(
        cell(h, True, "DCE6F1", widths[i]) for i, h in enumerate(head))
        + "</w:tr>")
    for r in rows:
        sh = "FFF2CC" if (highlight and highlight(r)) else None
        out.append("<w:tr>" + "".join(
            cell(c, False, sh, widths[i]) for i, c in enumerate(r)) + "</w:tr>")
    out.append("</w:tbl>")
    out.append(para(" "))
    return "".join(out)


def build(body):
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f'<w:body>{body}'
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
            '</w:sectPr></w:body></w:document>')


# ------------------------------------------------------------------- content
def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x*100:.2f}"


def load_true():
    d = P / "Analysis1" / "TRUE"
    if not (d / "run_manifest.json").exists():
        return {}, {}
    man = json.loads((d / "run_manifest.json").read_text("utf-8"))
    return {f.stem: np.load(f) for f in sorted(d.glob("*.npy"))}, man


def load_json(name):
    f = P / "Optimized" / name
    return json.loads(f.read_text("utf-8")) if f.exists() else None


def main():
    b = []
    now = datetime.datetime.now()
    tr, man = load_true()
    v2, v3 = load_json("optimize_v2.json"), load_json("optimize_v3.json")
    wts, probe = load_json("optimize_weights.json"), load_json("feature_probe.json")
    p2 = load_json("paper2_model.json")

    # ---------------------------------------------------------------- title
    b.append(para("Paper 1 — Complete Work Record", "Title"))
    b.append(para("Design and Development of a Video Forgery Model Using Deep "
                  "Learning with Attention Mechanisms (SMA-CLMPNet)", "Caption"))
    b.append(para(f"Generated {now:%Y-%m-%d %H:%M:%S} · Windows 11 · conda env "
                  f"VideoForgeryCPU (Python 3.8.20, TensorFlow 2.10, CPU only)",
                  "Caption"))

    # ------------------------------------------------------------- summary
    b.append(para("1. Executive Summary", "Heading1"))
    b.append(para("Three findings, in order of importance."))
    b.append(para("1.1 The reported metrics are fabricated", "Heading2"))
    b.append(para(
        "Every accuracy, sensitivity, specificity, precision, F1 and ROC value "
        "this codebase produces is a random number, independent of the model, "
        "the features and the training. The vendored mealpy/metrics.py has a "
        "modified _check_targets() that discards the classifier's predictions "
        "and replaces them with the ground-truth labels with a uniformly "
        "random fraction of entries flipped. A perfect predictor scores "
        "between 64.5% and 100.0% across repeated calls; an inverted predictor "
        "scores no worse; two identical calls disagree."))
    b.append(para("1.2 Paper 2's 100% is the maximum of 500 random draws",
                  "Heading2"))
    b.append(para(
        "Paper 2's SubFunctions/Optimization.py receives the test features and "
        "test labels in its constructor and searches model weights to maximise "
        "the score computed on them. That score is the fabricated metric, and "
        "HYBRID(epoch=10, pop_size=50) performs 500 evaluations keeping the "
        "best. Reproduced here: 30 of 30 independent runs return exactly "
        "100.00%. The same tampered file is present in the Paper 2 delivery, "
        "on the identical code path."))
    b.append(para("1.3 The features carry real but modest temporal signal",
                  "Heading2"))
    b.append(para(
        "Measured with a correct confusion matrix, the proposed SMA-CLMPNet "
        "achieves 50.00% balanced accuracy — it assigns one label to every "
        "video. However the 'proposed' feature tensor does carry signal: "
        "first-order temporal deltas reach 77.17% balanced accuracy under "
        "nested cross-validation, permutation-tested at p = 0.0099. The "
        "feature design is the paper's genuine contribution; the model fails "
        "to extract from it."))

    # ------------------------------------------------------- the fabrication
    b.append(para("2. The Fabricated Metric — Evidence", "Heading1"))
    b.append(para("2.1 The tampered code", "Heading2"))
    b.append(para("mealpy/metrics.py, lines 16–75, inside _check_targets(), "
                  "immediately before the confusion matrix is computed:"))
    b.append(code([
        "if perf:",
        "    per = random.uniform(0.065242, 0.35245235634)",
        "else:",
        "    per = random.uniform(0.090242, 0.45245235634)",
        "...",
        "y = np.concatenate(dat)",
        "y_true = shuffle(y, random_state=0)",
        "y_pred = y_true.copy()                       # predictions discarded",
        "va = random.sample(range(1, len(y_true)), int(len(y_true) * per))",
        "for i in va:",
        "    y_pred[i] = (random.sample(range(0, n), 1))[0]   # randomised",
    ]))
    b.append(para("Upstream mealpy contains no such code. The genuine function "
                  "survives, commented out, at mealpy/metrics.py:285."))
    b.append(para("2.2 Demonstration", "Heading2"))
    b.append(para("SubFunctions.Evaluate.Evaluation_Metrics as delivered, "
                  "y_true fixed at 15 zeros and 16 ones:"))
    b.append(table(
        ["Predictor", "True accuracy", "Reported (3 consecutive calls)"],
        [["Perfect", "1.000", "0.839, 0.935, 0.806"],
         ["Inverted (all wrong)", "0.000", "0.871, 0.710, 0.968"],
         ["Constant, all class 0", "0.484", "0.774, 0.774, 0.935"],
         ["Uniform random", "0.581", "0.806, 0.935, 0.839"]],
        [2600, 1800, 3600]))
    b.append(para("400 calls with a perfect predictor: min 0.645, max 1.000, "
                  "mean 0.878. A correct metric returns 1.000 every time."))
    b.append(para("2.3 Blast radius", "Heading2"))
    b.append(table(["Affected", "Route"],
                   [["§5.6.1 comparison vs training percentage",
                     "Analysis.py:184 → Evaluation_Metrics"],
                    ["§5.6.2 k-fold comparison",
                     "Analysis.py:233 → Evaluation_Metrics"],
                    ["§5.8 statistical analysis", "derived from same arrays"],
                    ["All ROC curves",
                     "Analysis.py:286-291 → Evaluation_Metrics1"],
                    ["Analysis/ and Analysis1/ .npy arrays", "same"],
                    ["Paper 2 (CODE_05-08-2025_Paper2)",
                     "SubFunctions/Evaluate.py:1, calls at lines 18 and 53"]],
                   [4200, 4600]))
    b.append(para("2.4 The fix", "Heading2"))
    b.append(para(
        "Optimized/metrics_fixed.py reimplements the metrics. The formulas in "
        "SubFunctions/Evaluate.py are correct as written, including the "
        "convention that class 0 is positive (TP = cm[0,0]); they are "
        "reproduced verbatim apart from zero-division guards. Only the "
        "confusion matrix is replaced, with sklearn's. Nothing in "
        "SubFunctions/ or mealpy/ was edited — the tampered code remains in "
        "place as evidence. Self-test: perfect 1.0000, inverted 0.0000, "
        "all-zeros 0.4839, deterministic over 50 calls."))

    # ------------------------------------------------------- Paper 2 finding
    b.append(para("3. Paper 2 Reference Study", "Heading1"))
    b.append(para("3.1 What Optimization.py does", "Heading2"))
    b.append(code([
        "class Optimization:",
        "    def __init__(self, model, x_test, y_test):   # receives TEST data",
        "        ...",
        "    def fitness_function1(self, solution):",
        "        self.model.set_weights(to_opt)",
        "        ypred = np.argmax(self.model.predict(self.x_test), axis=1)",
        "        A = Evaluation_Metrics(self.y_test, ypred)  # scores on TEST",
        "        return A[0]                                 # ...maximises it",
        "",
        "    model = HYBRID(epoch=10, pop_size=50)   # 500 evaluations, keep max",
    ]))
    b.append(para("3.2 Reproduction", "Heading2"))
    b.append(table(["500 fitness evaluations", "Value"],
                   [["Minimum", "67.74%"], ["Mean", "87.29%"],
                    ["Maximum — what solve() returns", "100.00%"],
                    ["Independent runs reaching 100%", "30 / 30"]],
                   [5000, 2500]))
    b.append(para("3.3 Paper 2's model applied to Paper 1", "Heading2"))
    b.append(para(
        "BiLSTMGBM was ported in full — stacked Bi-LSTM (100/128/128), "
        "multi-level and mixed attention, incremental learning over 5 "
        "cumulative chunks, 500 epochs, batch 32, lr 0.001, and the network "
        "used as a feature extractor with GradientBoosting on top — omitting "
        "only the test-set fitting step. Stratified 5-fold, correct scoring:"))
    if p2:
        g, o = p2["BiLSTMGBM"], p2["BiLSTM_only"]
        b.append(table(
            ["Configuration", "ACC", "SEN", "SPE", "PRE", "F1", "BAL"],
            [["BiLSTMGBM @ 500 epochs"] + [pct(x) for x in g],
             ["BiLSTM softmax only"] + [pct(x) for x in o]],
            [3000] + [900] * 6))
    b.append(para("Paper 2's architecture on Paper 1's data lands at chance "
                  "once the test-set fitting is removed."))

    # ------------------------------------------------------------- steps
    b.append(para("4. Step-by-Step Record", "Heading1"))
    steps = [
        ("Environment", "conda env VideoForgeryCPU (Python 3.8.20, TF 2.10, "
         "keras 2.10, numpy 1.21.6). Requires <env>\\Library\\bin on PATH or "
         "scipy triggers a DLL delay-load crash (0xc06d007f)."),
        ("Dataset", "FaceForensics++ is form-gated and could not be downloaded "
         "programmatically. Evaluation used Features/Features.pkl, which holds "
         "pre-extracted features for 50 videos (29 authentic / 21 forged)."),
        ("torch bypass", "SubFunctions/__init__.py imports torch, which ships a "
         "libiomp5md.dll colliding with conda-forge scipy's. SubFunctions is "
         "registered as a namespace package so __init__.py never runs."),
        ("BA-TFD excluded", "Its ViTDCNN applies MaxPooling2D(1,1), which does "
         "not downsample, so the flattened 1,048,576-element vector entering "
         "Dense(2048) needs an 8.6 GB weight matrix. OOM at every batch size."),
        ("KFAnalysis defect", "Analysis.py:355 indexes data['image'], a key "
         "ReadDataset never stores. K-fold was reimplemented with "
         "StratifiedKFold."),
        ("Metric fabrication found", "Four unrelated backbones returned "
         "byte-identical scores on all six splits while their predictions "
         "disagreed on 12–19 of 31 test samples. Traced to mealpy."),
        ("Corrected re-evaluation", "All seven of the paper's models re-run "
         "across six training percentages with a real confusion matrix "
         "(10,109 s)."),
        ("Modern architectures", "EfficientNetV2-S, ConvNeXt-Tiny, "
         "MobileNetV3-Large, ResNet-RS-50 as frozen ImageNet extractors — "
         "fine-tuning 5–25 M parameters on 19–44 samples would memorise."),
        ("Recipe optimisation", "SMA-CLMPNet retrained with batch 32→8, "
         "training-split input standardisation, class weights and cosine LR "
         "decay; architecture untouched."),
        ("Representation search", "19 representations × 14 model families "
         "under nested cross-validation, permutation-tested."),
        ("Weight/threshold sweep", "30 configurations of class weights, "
         "probability calibration and decision threshold, all selected inside "
         "training folds."),
        ("FF++ pipeline", "Frame-level training pipeline built and smoke-"
         "tested end to end, ready for the real dataset."),
    ]
    b.append(table(["Step", "Detail"], steps, [2400, 6400]))

    # ----------------------------------------------------------- results
    b.append(para("5. Results", "Heading1"))
    if tr:
        splits = ", ".join(f"{p:.0%}" for p in man["train_pcts"])
        b.append(para("5.1 Claimed vs measured", "Heading2"))
        b.append(para(f"Same models, same splits ({splits}). Only the scoring "
                      f"differs."))
        fab = {"EfficientNet": 89.00, "STIDNet": 85.34, "DCNN": 91.56,
               "GLCM": 92.19, "MUSE-CLMPNet": 89.72, "SCAM-CLMPNet": 90.47,
               "SMA-CLMPNet": 92.11}
        rows = []
        for n, f in fab.items():
            if n in tr:
                a = tr[n]
                rows.append([n, f"{f:.2f}", pct(np.nanmean(a[:, 0])),
                             pct(np.nanmean(a[:, 5])),
                             f"{np.nanmean(a[:,0])*100-f:+.2f}"])
        b.append(table(["Model", "Claimed", "Measured ACC", "Measured BAL",
                        "Delta"], rows, [2400, 1400, 1700, 1700, 1300],
                       highlight=lambda r: r[0] == "SMA-CLMPNet"))

        b.append(para("5.2 All models, mean over six splits", "Heading2"))
        rows = []
        for n in sorted(tr, key=lambda k: -np.nanmean(tr[k][:, 5])):
            a = tr[n]
            g = lambda i: ("n/a" if np.all(np.isnan(a[:, i]))
                           else pct(np.nanmean(a[:, i])))
            rows.append([n, g(0), g(1), g(2), g(3), g(4), g(5)])
        b.append(table(["Model", "ACC", "SEN", "SPE", "PRE", "F1", "BAL"],
                       rows, [2600] + [1000] * 6,
                       highlight=lambda r: r[6] in ("50.00", "n/a")))
        b.append(para("Highlighted rows are at exactly 50.00% balanced "
                      "accuracy: the model assigns one label to every video. "
                      "Their 50–56% accuracy figures are the 29/21 class "
                      "ratio, not discrimination.", "Caption"))

        b.append(para("5.3 Accuracy by training percentage", "Heading2"))
        hdr = ["Model"] + [f"{int(p*100)}%" for p in man["train_pcts"]] + ["Mean"]
        rows = []
        for n in sorted(tr, key=lambda k: -np.nanmean(tr[k][:, 0])):
            a = tr[n][:, 0]
            rows.append([n] + [pct(x) for x in a] + [pct(np.nanmean(a))])
        b.append(table(hdr, rows, [2400] + [900] * (len(hdr) - 1)))

    # ------------------------------------------------- optimisation results
    b.append(para("6. Optimisation Attempts", "Heading1"))
    b.append(para("6.1 Representation and model search", "Heading2"))
    b.append(para("Nested cross-validation throughout: hyper-parameters are "
                  "chosen inside each outer fold, so the reported score is "
                  "estimated on data the selection never saw."))
    if v2:
        rows = [[f"{r['mean']*100:.2f}", f"±{r['std']*100:.2f}", r["model"],
                 r["representation"]] for r in v2["ranking"][:12]]
        b.append(table(["BAL", "SD", "Model", "Representation"], rows,
                       [900, 900, 1600, 5000],
                       highlight=lambda r: "temporal delta" in r[3]))
        w = v2["winner"]
        b.append(para(
            f"Permutation test on the winner: observed "
            f"{w['nested_bal_acc']*100:.2f}%, null mean "
            f"{w['null_mean']*100:.2f}%, null 95th percentile "
            f"{w['null_p95']*100:.2f}%, p = {w['p_value']:.4f}."))
    b.append(para("6.2 Higher-order features and ensembles", "Heading2"))
    if v3:
        rows = [[f"{r['mean']*100:.2f}", f"±{r['std']*100:.2f}", r["model"],
                 r["representation"]] for r in v3["ranking"][:8]]
        b.append(table(["BAL", "SD", "Model", "Representation"], rows,
                       [900, 900, 1600, 5000]))
    b.append(para("Acceleration, lag-2 differences, autocorrelation, soft "
                  "voting, stacking and feature fusion all scored BELOW plain "
                  "L1 logistic regression on first-order deltas. At n = 50, "
                  "added complexity buys variance, not accuracy."))
    b.append(para("6.3 Class-weight and threshold optimisation", "Heading2"))
    if wts:
        rows = [[f"{r['bal_acc']*100:.2f}", f"±{r['std']*100:.2f}",
                 f"{r['threshold']:.2f}", r["model"]]
                for r in wts["ranking"][:8]]
        b.append(table(["BAL", "SD", "Threshold", "Configuration"], rows,
                       [900, 900, 1200, 5400]))
    b.append(para("30 configurations of class weights, calibration and "
                  "decision threshold, all selected on training folds. Best "
                  "69.67% — below the 77.17% of the untuned linear model."))
    b.append(para("6.4 Signal test", "Heading2"))
    b.append(table(["Protocol", "Balanced accuracy", "p-value"],
                   [["Nested 5-fold stratified CV", "77.17%", "0.0099"],
                    ["Repeated random stratified splits", "62.60%", "—"],
                    ["The paper's prefix split", "55.20%", "—"]],
                   [4000, 2400, 1600]))
    b.append(para("The paper's split takes the first N of each class, so any "
                  "ordering in the data becomes a train/test distribution "
                  "shift. Stratified splits recover roughly 7 points."))

    b.append(para("7. Training / Validation / Test Accuracy", "Heading1"))
    b.append(table(
        ["Split", "Train ACC", "Validation (inner CV)", "Test ACC", "Test BAL"],
        [["40%", "100.00", "64.58", "58.06", "57.48"],
         ["50%", "87.50", "60.42", "42.31", "41.52"],
         ["60%", "100.00", "70.00", "61.90", "62.50"],
         ["70%", "91.18", "78.75", "62.50", "61.90"],
         ["80%", "87.18", "77.08", "36.36", "38.33"],
         ["90%", "100.00", "73.72", "66.67", "66.67"],
         ["Mean", "94.31", "70.76", "54.63", "54.73"]],
        [1400, 1800, 2600, 1700, 1700],
        highlight=lambda r: r[0] == "Mean"))
    b.append(para("Training accuracy reaches 100% on three of six splits: with "
                  "324 features and 19 training samples a separating "
                  "hyperplane always exists. The 94.31% → 70.76% → 54.63% "
                  "cascade is overfitting driven by sample count. Note that a "
                  "95–99% figure is obtainable simply by reporting training "
                  "accuracy."))

    # ------------------------------------------------------------- FF++
    b.append(para("8. FaceForensics++ Pipeline", "Heading1"))
    b.append(para("Built, smoke-tested, awaiting the dataset. This is the only "
                  "route to genuine 90%-range accuracy."))
    b.append(table(["", "Current evaluation", "FF++ pipeline"],
                   [["Videos", "50", "~2,000"],
                    ["Training unit", "one vector per video", "face crop per frame"],
                    ["Training samples", "19–44", "~50,000–64,000"],
                    ["Test samples", "6–31 videos", "200–400 videos"]],
                   [2000, 3000, 3000]))
    b.append(para("8.1 Split integrity", "Heading2"))
    b.append(para(
        "FF++ names manipulated clips <target>_<source>.mp4, sharing footage "
        "with <target>.mp4. Splitting by frame — or even by video — lets the "
        "model recognise the footage rather than the manipulation, which "
        "drives accuracy toward 100% while measuring nothing. Splits are "
        "grouped by source identity and assert_disjoint() halts the run on any "
        "overlap. Stated in advance: low-to-mid 90s is consistent with the "
        "literature; near 99–100% should be treated as a leakage bug."))
    b.append(para("8.2 Usage", "Heading2"))
    b.append(code([
        "# 1. request access at github.com/ondyari/FaceForensics",
        "python faceforensics_download_v4.py DATASET -d original -c c23 -t videos",
        "python faceforensics_download_v4.py DATASET -d FaceSwap -c c23 -t videos",
        "",
        "# 2. one command: preflight -> ingest -> baseline -> report",
        "python FFPP/run_baseline.py --root DATASET --backbone EfficientNetV2B0",
        "",
        "# verify the pipeline before the data arrives",
        "python FFPP/smoke_test.py",
    ]))

    # -------------------------------------------------------- conclusions
    b.append(para("9. Conclusions", "Heading1"))
    for t in [
        "The reported metrics in both Paper 1 and Paper 2 are fabricated by a "
        "modified mealpy/metrics.py and cannot be used.",
        "Paper 2's 100% accuracy is the maximum of 500 random draws obtained "
        "by fitting model weights to the test set; it reproduces 30/30 times "
        "and is not transferable.",
        "Measured correctly, SMA-CLMPNet achieves 50.00% balanced accuracy — a "
        "constant prediction — and ranks 6th of 12 on its own benchmark, "
        "below an off-the-shelf MobileNetV3 (59.75%).",
        "The 'proposed' feature tensor does carry temporal signal: 77.17% "
        "balanced accuracy at p = 0.0099. The feature design is the genuine "
        "contribution; the model does not exploit it.",
        "A linear model on 324 temporal statistics outperforms the full 3D-CNN "
        "+ dual-LSTM + attention architecture by roughly 27 points.",
        "Architecture generation is uncorrelated with performance here: "
        "MobileNetV3-Large (2019) beats ConvNeXt-Tiny (2022) and ResNet-RS-50.",
        "The binding constraint is 19–44 training samples. 95–99% accuracy is "
        "unreachable on this corpus by any legitimate configuration.",
        "The route to publishable accuracy is the real FaceForensics++ data, "
        "three to four orders of magnitude larger. The pipeline is ready.",
    ]:
        b.append(bullet(t))

    b.append(para("10. Recommendations", "Heading1"))
    for t in [
        "Do not submit or circulate either paper while the results sections "
        "contain figures no model produced.",
        "Replace mealpy/metrics.py, or route all scoring through "
        "Optimized/metrics_fixed.py, before any further experiment.",
        "Report balanced accuracy and the confusion matrix alongside accuracy: "
        "on a 29/21 corpus a constant classifier scores 58%.",
        "Use stratified splits, not the first-N-per-class prefix split.",
        "Acquire FaceForensics++ and train at frame level.",
        "Consider reframing the contribution around the temporal feature "
        "design, which is defensible and supported by evidence.",
    ]:
        b.append(bullet(t))

    # ---------------------------------------------------------- inventory
    b.append(para("11. File Inventory", "Heading1"))
    inv = [
        ("Optimized/metrics_fixed.py", "Correct metrics; carries a self-test"),
        ("Optimized/optimize_models.py", "Re-runs the paper's models with "
         "correct scoring; adds modern backbones and the optimised recipe"),
        ("Optimized/optimize_v2.py", "19-representation × 14-model nested-CV "
         "search with permutation test"),
        ("Optimized/optimize_v3.py", "Higher-order temporal features and "
         "stacked ensembles"),
        ("Optimized/optimize_weights.py", "Class-weight, calibration and "
         "decision-threshold sweep"),
        ("Optimized/feature_probe.py", "Independent signal probe"),
        ("Optimized/paper2_model.py", "Paper 2's BiLSTMGBM ported, minus the "
         "test-set fitting"),
        ("Optimized/frame_embeddings.py", "Per-frame and temporal-delta "
         "backbone embeddings"),
        ("Optimized/final_tables.py", "Full metric tables by training "
         "percentage"),
        ("Optimized/report.py", "Builds RESULTS.md"),
        ("Optimized/correct_doc.py", "Rewrites §5.6.1/§5.6.2/§5.8 from "
         "measured arrays"),
        ("Optimized/INTEGRITY_FINDING.md", "Evidence for the fabricated metric"),
        ("Optimized/COMPARISON.md", "Comparison with other work"),
        ("FFPP/ffpp_data.py", "FF++ ingestion: videos → cached face crops"),
        ("FFPP/ffpp_train.py", "Frame-level training, video-level evaluation"),
        ("FFPP/run_baseline.py", "One-command preflight → ingest → baseline"),
        ("FFPP/smoke_test.py", "End-to-end verification on synthetic videos"),
    ]
    b.append(table(["File", "Purpose"], [list(x) for x in inv], [3400, 5400]))
    b.append(para("Nothing in SubFunctions/ or mealpy/ was modified. The "
                  "tampered code remains in place as evidence and every "
                  "correction is additive.", "Caption"))

    # ------------------------------------------------------------- write
    doc = build("".join(b))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", doc)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)
        z.writestr("word/styles.xml", STYLES)
    print(f"wrote {OUT.name}  ({OUT.stat().st_size/1024:.0f} KB)")

    # validate
    with zipfile.ZipFile(OUT) as z:
        bad = z.testzip()
        assert bad is None, bad
        import xml.dom.minidom as md
        for part in ("word/document.xml", "word/styles.xml",
                     "[Content_Types].xml"):
            md.parseString(z.read(part))
    paras = doc.count("<w:p>")
    tables = doc.count("<w:tbl>")
    print(f"validated: zip OK, XML well-formed, {paras} paragraphs, "
          f"{tables} tables")


if __name__ == "__main__":
    main()
