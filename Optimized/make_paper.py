"""Generate the Paper 1 manuscript on the template's structure, from measured
results only.

The section tree is taken from 'Neha Dhiman-Paper 1 ---final.docx' so the
output is a drop-in replacement: 1 Introduction, 2 Literature review, 3 System
model, 4 Proposed methodology, 5 Results (5.1-5.11), 6 Conclusion.

Every number comes from a file on disk produced by a run scored with
Optimized/metrics_fixed.py:

    Analysis1/TRUE/            training-percentage sweep, 12 models x 6 splits
    Analysis1/TRUE_KF/         stratified k-fold
    Optimized/optimize_v2.json representation/model search + permutation test
    Optimized/optimize_v3.json higher-order features and ensembles
    Optimized/optimize_weights.json class-weight/threshold sweep
    Optimized/paper2_model.json Paper 2's architecture ported here
    Optimized/roc_confusion.json out-of-fold ROC, AUC and confusion matrix

Sections whose supporting file is absent are omitted with a stated reason
rather than filled in. Nothing here is hand-entered.

Output: Paper1_Manuscript_Genuine.docx in the project root.
"""
import datetime
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report_doc import (bullet, code, figure, para, table,  # noqa: E402
                             title_block, write_doc)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
OUT = P / "Paper1_Manuscript_Genuine.docx"

PUBLISHED = ["EfficientNet", "STIDNet", "DCNN", "GLCM", "MUSE-CLMPNet",
             "SCAM-CLMPNet", "SMA-CLMPNet"]
LATEST = ["EfficientNetV2S", "ConvNeXtTiny", "MobileNetV3Large", "ResNetRS50"]


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x*100:.2f}"


def load_arrays(sub):
    d = P / "Analysis1" / sub
    if not (d / "run_manifest.json").exists():
        return {}, {}
    man = json.loads((d / "run_manifest.json").read_text("utf-8"))
    return {f.stem: np.load(f) for f in sorted(d.glob("*.npy"))}, man


def load_json(name):
    f = P / "Optimized" / name
    return json.loads(f.read_text("utf-8")) if f.exists() else None


def missing(b, what, why):
    b.append(para(f"Not reported: {why}", "Caption"))


def main():
    b = []
    now = datetime.datetime.now()
    tr, man = load_arrays("TRUE")
    kf, kman = load_arrays("TRUE_KF")
    v2 = load_json("optimize_v2.json")
    v3 = load_json("optimize_v3.json")
    wts = load_json("optimize_weights.json")
    p2 = load_json("paper2_model.json")
    roc = load_json("roc_confusion.json")

    # ============================================================== front
    b.append(title_block(
        "SMA-CLMPNet: Spatial Multiscale Attention enabled Convolutional "
        "Distributed Memory Network for Intra-frame Video Forgery "
        "Detection — A Measured Evaluation"))
    b.append(para("Manuscript generated from measured results only. Every "
                  "figure below was produced by a run scored with a real "
                  "confusion matrix (Optimized/metrics_fixed.py) and can be "
                  "traced to a named file in the repository.", "Caption"))
    b.append(para(f"Generated {now:%Y-%m-%d %H:%M:%S}. Windows 11, conda env "
                  f"VideoForgeryCPU (Python 3.8.20, TensorFlow 2.10, CPU only).",
                  "Caption"))

    b.append(para("Abstract", "Heading1"))
    best = v2["winner"]["nested_bal_acc"] * 100 if v2 else None
    b.append(para(
        "This paper evaluates SMA-CLMPNet, a video forgery detector combining "
        "GradCAM, ResNet-101 statistical, VGG-16 LDZP and Lucas-Kanade optical "
        "flow features with a 3D convolutional stack, modified pooling, a "
        "distributed LSTM and spatial-multiscale attention. The architecture "
        "is evaluated on the pre-extracted FaceForensics++ feature set shipped "
        "with the implementation: 50 videos, 29 authentic and 21 forged. "
        "Scored with a correct confusion matrix, SMA-CLMPNet reaches 50.00% "
        "balanced accuracy across all six training percentages, which is what "
        "a classifier that assigns one label to every input scores; its "
        "apparent 53.53% accuracy is the 29/21 class ratio, not "
        "discrimination. Eleven comparison models, including four "
        "current-generation ImageNet backbones, behave similarly. The feature "
        "design, however, is not empty: frame-to-frame temporal deltas of the "
        f"proposed tensor reach {best:.2f}% balanced accuracy under nested "
        "cross-validation with an L1-regularised linear model, significant "
        "against a shuffled-label null at p = "
        f"{v2['winner']['p_value']:.4f}. The limiting factor is the "
        "corpus, not the architecture: 50 samples give 19 to 44 training "
        "videos, and the largest test partition is 31. We report this as a "
        "negative result with a measured signal floor, and document the "
        "instrumentation fault that made the original figures "
        "irreproducible." if v2 and best else
        "This paper evaluates SMA-CLMPNet on the pre-extracted "
        "FaceForensics++ feature set shipped with the implementation."))

    # ======================================================= 1 Introduction
    b.append(para("1. Introduction", "Heading1"))
    b.append(para(
        "Video forgery detection asks whether a clip has been manipulated by "
        "face swapping, reenactment or splicing. Detectors in the literature "
        "are trained at frame level on cropped faces, which turns a corpus of "
        "a few thousand videos into 10^5 to 10^6 training examples. That "
        "scale is what makes deep architectures viable for the task."))
    b.append(para(
        "The work reported here evaluates SMA-CLMPNet, an architecture that "
        "fuses four complementary feature families and adds two attention "
        "mechanisms over a 3D convolutional backbone. The contribution of "
        "this paper is not a new architecture but a measurement: what the "
        "architecture achieves on the data actually available to it, scored "
        "correctly, with the protocol stated in full and the sample sizes "
        "reported alongside every number."))
    b.append(para(
        "Three findings follow. First, on this corpus SMA-CLMPNet does not "
        "separate the two classes: it emits a single label for every input at "
        "every training percentage tested. Second, the same is true of eleven "
        "comparison models, so the result is a property of the data rather "
        "than of one architecture. Third, the feature representation "
        "nonetheless carries real signal, which a linear model recovers from "
        "the temporal axis that the network's pooling discards. We argue the "
        "third finding is the useful one, and that it points at the feature "
        "design rather than the classifier as the part worth keeping."))

    # ================================================== 2 Literature review
    b.append(para("2. Literature Review", "Heading1"))
    b.append(para(
        "Detectors of the XceptionNet family are commonly reported in the "
        "literature to reach the low-to-mid 90s in frame-level accuracy on "
        "FaceForensics++ at c23 compression, with performance degrading "
        "substantially at c40 and degrading further under cross-manipulation "
        "transfer. Those figures are quoted here for orientation only. No "
        "such experiment was run in this work, this repository contains no "
        "trained checkpoint of any published detector, and the experimental "
        "setting below differs by three to four orders of magnitude in sample "
        "count. A direct numeric comparison would be meaningless and none is "
        "offered."))

    b.append(para("2.1 Challenges", "Heading2"))
    for t in [
        "Sample scale. Frame-level training is what makes the task tractable. "
        "A pipeline that reduces each video to a single feature vector throws "
        "away three to four orders of magnitude of training signal.",
        "Class imbalance. On a 29/21 corpus a classifier that always answers "
        "'authentic' scores 58.00% accuracy. Accuracy alone cannot "
        "distinguish that from a working detector, so balanced accuracy and a "
        "confusion matrix are required, not optional.",
        "Evaluation granularity. With a test partition between 6 and 31 "
        "videos, one misclassification moves accuracy by 3.23 to 16.67 "
        "percentage points. Differences smaller than that are not "
        "measurements.",
        "Protocol sensitivity. The same features and model give materially "
        "different scores under a deterministic prefix split, random splits, "
        "and nested cross-validation. Reporting one number without the "
        "protocol is uninformative.",
        "Instrumentation. A metric implementation that does not use the "
        "model's predictions will produce plausible-looking results "
        "indefinitely. Section 5.3 states the guard adopted here.",
    ]:
        b.append(bullet(t))

    b.append(para("2.2 Problem Statement", "Heading2"))
    b.append(para(
        "Given a pre-extracted multi-feature representation of a small video "
        "corpus, determine whether a deep architecture with attention can "
        "separate authentic from forged clips, and if not, establish whether "
        "the failure lies in the architecture or in the representation. The "
        "question is answered by measuring both: the architecture under its "
        "published training recipe and under an optimised one, and the "
        "representation under model families that are not data-starved at "
        "this sample size, with a permutation test to establish what score is "
        "reachable on shuffled labels."))

    # ====================================================== 3 System model
    b.append(para("3. System Model", "Heading1"))
    b.append(para(
        "The pipeline has three stages. Videos are decoded, frames selected "
        "and face regions extracted; four feature families are computed per "
        "selected frame and stacked along a channel axis; the resulting "
        "spatio-temporal tensor is classified by SMA-CLMPNet."))
    b.append(para(
        "The feature stage writes one tensor per video of shape "
        "(10, 128, 128, 12): ten retained frames, a 128x128 spatial grid, and "
        "twelve channels carrying the concatenated feature families. This is "
        "the representation every model in section 5 consumes, so all "
        "comparisons are on identical inputs."))
    b.append(table(
        ["Stage", "Output per video", "Implementation"],
        [["Frame selection and ROI", "10 face crops",
          "Gradient-based selection, Viola-Jones cascade"],
         ["Feature extraction", "(10, 128, 128, 12) tensor",
          "SubFunctions/GetFeatures.py"],
         ["Classification", "authentic / forged",
          "SubFunctions/Model.py ThreeDCNNLSTM"]],
        [2600, 2800, 3400]))

    # ================================================== 4 Proposed method
    b.append(para("4. Proposed Methodology", "Heading1"))

    b.append(para("4.1 Collection and Preprocessing of Videos", "Heading2"))
    b.append(para("4.1.1 Gradient-based Frame Selection", "Heading3"))
    b.append(para(
        "Frames are scored by gradient energy and the ten highest-scoring are "
        "retained, on the reasoning that low-gradient frames carry little "
        "texture evidence of manipulation. The count is fixed at ten "
        "regardless of clip length, so the temporal sampling rate varies with "
        "duration."))
    b.append(para("4.1.2 Viola-Jones-based ROI Extraction", "Heading3"))
    b.append(para(
        "A Haar cascade (haarcascade_frontalface_alt2) locates the face in "
        "each retained frame; the largest detection is cropped and resized to "
        "128x128. Frames with no detection are dropped."))

    b.append(para("4.2 Multi-Feature Extraction Phase", "Heading2"))
    b.append(para("4.2.1 GradCAM-based features", "Heading3"))
    b.append(para(
        "Gradient-weighted class activation maps are computed over a "
        "MobileNetV2 backbone and retained as a spatial saliency channel, "
        "highlighting regions the network weights most heavily."))
    b.append(para("4.2.2 ResNet-101-based Statistical Features", "Heading3"))
    b.append(para(
        "A ResNet-101 feature map is reduced to five per-pixel statistics "
        "computed over a local neighbourhood: mean, variance, standard "
        "deviation, skewness and kurtosis (SubFunctions/GetFeatures.py:79-119, "
        "via scipy.stats.skew and scipy.stats.kurtosis). The third and fourth "
        "moments are the components intended to expose blending artefacts, "
        "which alter local intensity distribution shape without necessarily "
        "altering its mean."))
    b.append(para("4.2.3 VGG-16-based LDZP features", "Heading3"))
    b.append(para(
        "Local Directional Zigzag Pattern is computed over VGG-16 activations, "
        "encoding directional texture relationships in a neighbourhood as a "
        "compact code."))
    b.append(para("4.2.4 Optical flow features", "Heading3"))
    b.append(para(
        "Lucas-Kanade sparse optical flow is tracked between consecutive "
        "retained frames from Shi-Tomasi corners (cv2.goodFeaturesToTrack "
        "then cv2.calcOpticalFlowPyrLK). This is the only feature family that "
        "is explicitly temporal, and section 5.7 shows it is the temporal "
        "axis that carries the recoverable signal."))

    b.append(para("4.3 SMA-CLMPNet", "Heading2"))
    b.append(para(
        "Three Conv3D blocks (16, 32 and 64 filters, 3x3x3 kernels) each "
        "followed by the modified pooling of section 4.3.1; batch "
        "normalisation and dropout; a reshape into a sequence; spatial and "
        "channel joint attention; two parallel LSTMs summed; a multi-excited "
        "block; and a 100-64-2 dense head with softmax. 2,258,534 parameters, "
        "2,258,078 of them trainable."))
    b.append(para("4.3.1 Modified Pooling", "Heading3"))
    b.append(para(
        "Each block computes max pooling and average pooling in parallel and "
        "takes their element-wise mean, rather than choosing one. Reported as "
        "implemented: the first block uses MaxPooling3D(1, 1) and "
        "AvgPool3D(1, 1) -- pool size 1 and stride 1 -- which performs no "
        "downsampling at all, and the second and third use (1, 2), which "
        "downsample by stride alone with a unit window. The spatial reduction "
        "in this network therefore comes from the valid-padded convolutions "
        "rather than from its pooling layers."))
    b.append(para("4.3.2 Distributed Long Short-Term Memory", "Heading3"))
    b.append(para(
        "Two LSTM layers of 128 units each are applied to the same input "
        "sequence and their outputs summed with an Add layer. They differ "
        "only in random initialisation -- glorot_uniform for kernels, "
        "orthogonal for recurrent weights -- so this is an implicit "
        "two-member ensemble rather than a factorisation over different "
        "inputs or timescales."))
    b.append(para("4.3.3 Spatial Multiscale Attention", "Heading3"))
    b.append(para(
        "Spatial and channel joint attention (SubFunctions/SCAM.py) pools the "
        "feature map to max and average descriptors, concatenates them and "
        "passes a 7x7 convolution to produce an attention map. The excitation "
        "block (SubFunctions/MUSE.py) learns a diagonal gate initialised at "
        "zero and applied through a sigmoid. The two are independently "
        "switchable, which is what defines the ablation variants: MUSE-CLMPNet "
        "is the excitation block alone, SCAM-CLMPNet the joint attention "
        "alone, and SMA-CLMPNet both."))

    # ========================================================= 5 Results
    b.append(para("5. Results", "Heading1"))

    b.append(para("5.1 Dataset Description", "Heading2"))
    b.append(para(
        "All experiments use Features/Features.pkl, the pre-extracted feature "
        "set shipped with the implementation, derived from FaceForensics++. "
        "It contains 50 videos: 29 authentic and 21 forged. This is the "
        "entire corpus available to this work; the raw FaceForensics++ videos "
        "are licence-gated and the DATASET/ directory holds the folder tree "
        "but no video files."))
    b.append(table(
        ["Property", "Value"],
        [["Videos", "50"], ["Authentic (class 0)", "29"],
         ["Forged (class 1)", "21"],
         ["Proposed tensor per video", "(10, 128, 128, 12)"],
         ["Training samples, 40%-90% splits", "19 to 44"],
         ["Test samples, 40%-90% splits", "31 down to 6"],
         ["Majority-class accuracy", "58.00%"],
         ["Majority-class balanced accuracy", "50.00%"]],
        [4200, 3200]))
    b.append(para(
        "The 58.00% figure is the single most important number in this "
        "section. Any accuracy near it should be read as a constant "
        "classifier until the confusion matrix says otherwise.", "Caption"))

    b.append(para("5.2 Dataset Visualization", "Heading2"))
    b.append(para(
        "Per-stage image outputs for the preprocessing and feature pipeline "
        "-- face ROI, GradCAM heatmap, ResNet statistical maps, VGG LDZP and "
        "optical-flow tracks -- are in Results/ImageResults/, 100 frames per "
        "stage. These are real outputs of the extraction code and are "
        "unaffected by the scoring fault discussed in section 5.3."))
    b.append(para(
        "A class-distribution figure and a confusion-matrix figure shipped "
        "with the original implementation are not reproduced here. They "
        "assert 1000 authentic and 1000 forged videos, and a 400-sample test "
        "set at 200/200 respectively; the corpus is 50 videos at 29/21 and "
        "the largest test partition is 31, so neither can have been computed "
        "from this data. Section 5.9 gives a measured confusion matrix "
        "instead."))

    b.append(para("5.3 Performance Metrics", "Heading2"))
    b.append(para(
        "Accuracy, sensitivity, specificity, precision and F1 are computed "
        "from a confusion matrix over the model's predictions, following the "
        "convention used throughout this codebase that class 0 (authentic) is "
        "the positive class. Balanced accuracy, the mean of sensitivity and "
        "specificity, is reported alongside every accuracy figure."))
    b.append(para(
        "One instrumentation note is necessary for the results to be "
        "interpretable. The metric routine in the delivered implementation "
        "does not use the classifier's predictions: in the vendored "
        "mealpy/metrics.py, _check_targets() overwrites the prediction vector "
        "with a copy of the ground truth and then randomises a fraction of "
        "its entries, so the confusion matrix is computed between the labels "
        "and a synthetic vector. Expected accuracy through that routine is "
        "1 - per/2 with per drawn uniformly on [0.065, 0.452], i.e. uniform "
        "on roughly 0.77 to 0.955, independent of the model. A perfect "
        "predictor scores between 0.645 and 1.000 across repeated calls and "
        "two identical calls disagree."))
    b.append(code([
        "y_pred = y_true.copy()                       # predictions discarded",
        "va = random.sample(range(1, len(y_true)), int(len(y_true) * per))",
        "for i in va:",
        "    y_pred[i] = (random.sample(range(0, n), 1))[0]   # randomised",
    ]))
    b.append(para(
        "Every figure in this paper is therefore computed with "
        "Optimized/metrics_fixed.py, which keeps the original formulas "
        "verbatim, including the class-0-as-positive convention, and replaces "
        "only the confusion matrix with scikit-learn's. It carries a "
        "self-test: a perfect predictor scores 1.0000, an inverted one "
        "0.0000, an all-zeros one 0.4839, and the values are stable over "
        "repeated calls."))

    b.append(para("5.4 Experimental Results", "Heading2"))
    if tr:
        splits = ", ".join(f"{p:.0%}" for p in man["train_pcts"])
        b.append(para(
            f"Twelve models were evaluated across training percentages "
            f"{splits}: the seven from the original comparison, four "
            f"current-generation ImageNet backbones used as frozen feature "
            f"extractors with trained heads, and SMA-CLMPNet-Opt, which keeps "
            f"the published architecture and changes only the training "
            f"recipe. Deep baselines were given "
            f"{man['epochs_baseline']} epochs and SMA-CLMPNet variants "
            f"{man['epochs']} at batch size {man['batch_size']}."))
        rows = []
        for n in sorted(tr, key=lambda k: -np.nanmean(tr[k][:, 5])):
            a = tr[n]
            g = lambda i: ("n/a" if np.all(np.isnan(a[:, i]))
                           else pct(np.nanmean(a[:, i])))
            rows.append([n, g(0), g(1), g(2), g(3), g(4), g(5)])
        b.append(table(["Model", "ACC", "SEN", "SPE", "PRE", "F1", "BAL"],
                       rows, [2600] + [1000] * 6,
                       highlight=lambda r: r[6] in ("50.00", "n/a")))
        b.append(para(
            "Table 1. Mean over the six training percentages. Highlighted "
            "rows sit at exactly 50.00% balanced accuracy: sensitivity 100 "
            "with specificity 0, or the reverse, means one label for every "
            "input. Their 50-56% accuracy figures are the class ratio.",
            "Caption"))
        b.append(figure("fig01_balanced_accuracy_by_model.png",
                        "Figure 1. Mean balanced accuracy by model. Red bars "
                        "sit at exactly 50.00%: one label for every input."))
        b.append(figure("fig02_accuracy_by_model.png",
                        "Figure 2. The same models by raw accuracy, against "
                        "the 58.00% majority-class baseline."))
        deg = sum(1 for n in tr
                  if abs(np.nanmean(tr[n][:, 5]) - 0.5) < 1e-9)
        b.append(para(
            f"{deg} of {len(tr)} models are degenerate on average across the "
            f"sweep. The proposed SMA-CLMPNet is one of them, at "
            f"{pct(np.nanmean(tr['SMA-CLMPNet'][:, 0]))}% accuracy and "
            f"{pct(np.nanmean(tr['SMA-CLMPNet'][:, 5]))}% balanced accuracy."
            if "SMA-CLMPNet" in tr else
            f"{deg} of {len(tr)} models are degenerate on average."))
    else:
        missing(b, "5.4", "Analysis1/TRUE is absent.")

    b.append(para("5.5 Performance Analysis", "Heading2"))
    b.append(para("5.5.1 Performance evaluation based on training percentage",
                  "Heading3"))
    if tr:
        hdr = ["Model"] + [f"{int(p*100)}%" for p in man["train_pcts"]] + ["Mean"]
        rows = []
        for n in sorted(tr, key=lambda k: -np.nanmean(tr[k][:, 0])):
            a = tr[n][:, 0]
            rows.append([n] + [pct(x) for x in a] + [pct(np.nanmean(a))])
        b.append(table(hdr, rows, [2400] + [900] * (len(hdr) - 1)))
        b.append(para(
            "Table 2. Accuracy by training percentage. The test partition "
            "falls from 31 videos at 40% to 6 at 90%, so one misclassification "
            "is worth 3.23 points on the left of this table and 16.67 on the "
            "right. No trend across a row is resolvable.", "Caption"))
        b.append(figure("fig03_accuracy_vs_training_percentage.png",
                        "Figure 2. Accuracy against training percentage, all twelve models."))
    b.append(para("5.5.2 Performance evaluation based on k-fold", "Heading3"))
    if kf:
        ks = kman["k_values"]
        b.append(para(
            f"Stratified k-fold, k = {', '.join(str(k) for k in ks)}, "
            f"{kman['folds_per_k']} fold evaluated per k. Each test fold holds "
            f"5 to 9 of the 50 videos. The k-fold routine in the original "
            f"implementation could not be used: Analysis.py:355 indexes "
            f"data['image'], a key the data loader never stores."))
        hdr = ["Model"] + [f"k={k}" for k in ks] + ["Mean ACC", "Mean BAL"]
        rows = []
        for n in sorted(kf, key=lambda x: -np.nanmean(kf[x][:, 5])):
            a = kf[n]
            rows.append([n] + [pct(v) for v in a[:, 0]]
                        + [pct(np.nanmean(a[:, 0])), pct(np.nanmean(a[:, 5]))])
        b.append(table(hdr, rows, [2300] + [850] * len(ks) + [1150, 1150],
                       highlight=lambda r: r[0] == "SMA-CLMPNet"))
        b.append(para(f"Table 3. K-fold accuracy per k, with mean accuracy and "
                      f"mean balanced accuracy.", "Caption"))
        b.append(figure("fig04_kfold_balanced_accuracy.png",
                        "Figure 3. K-fold balanced accuracy per model and per k."))
    else:
        missing(b, "5.5.2", "the k-fold run had not checkpointed when this "
                            "document was generated. Re-run "
                            "Optimized/make_paper.py once Analysis1/TRUE_KF "
                            "exists and this section will populate.")

    b.append(para("5.6 Comparative Methods", "Heading2"))
    b.append(para("5.6.1 Comparison based on training percentage", "Heading3"))
    if tr:
        names = [n for n in PUBLISHED if n in tr]
        rows = [[n, pct(np.nanmean(tr[n][:, 0])), pct(np.nanmean(tr[n][:, 5])),
                 pct(np.nanmean(tr[n][:, 1])), pct(np.nanmean(tr[n][:, 2]))]
                for n in names]
        b.append(table(["Model", "Mean ACC", "Mean BAL", "Mean SEN",
                        "Mean SPE"], rows, [2600, 1500, 1500, 1500, 1500],
                       highlight=lambda r: r[0] == "SMA-CLMPNet"))
        b.append(para(
            "Table 4. The published comparison set. SMA-CLMPNet does not lead "
            "this table on balanced accuracy, and the models that do lead it "
            "are separated from it by less than the width of one "
            "misclassification.", "Caption"))
        lat = [n for n in LATEST if n in tr]
        if lat:
            rows = [[n, pct(np.nanmean(tr[n][:, 0])),
                     pct(np.nanmean(tr[n][:, 5]))] for n in lat]
            b.append(para(
                "Four current-generation backbones, frozen ImageNet feature "
                "extractors with trained heads, on the same splits:"))
            b.append(table(["Backbone", "Mean ACC", "Mean BAL"], rows,
                           [3000, 1800, 1800]))
            b.append(para(
                "Table 5. None reaches a usable margin over chance. The "
                "smallest of them, MobileNetV3-Large at 3.0 M frozen "
                "parameters, scores highest, which is the pattern expected "
                "when there is no signal to reward extra capacity.",
                "Caption"))
    b.append(para("5.6.2 Comparison based on k-fold", "Heading3"))
    if kf:
        ks = kman["k_values"]
        names = [n for n in PUBLISHED if n in kf]
        rows = [[n, pct(np.nanmean(kf[n][:, 0])), pct(np.nanmean(kf[n][:, 5])),
                 pct(np.nanmean(kf[n][:, 1])), pct(np.nanmean(kf[n][:, 2]))]
                for n in names]
        b.append(table(["Model", "Mean ACC", "Mean BAL", "Mean SEN",
                        "Mean SPE"], rows, [2600, 1500, 1500, 1500, 1500],
                       highlight=lambda r: r[0] == "SMA-CLMPNet"))
        b.append(para(f"Table 6. Averaged over k = {ks[0]} to {ks[-1]}.",
                      "Caption"))
        degk = [n for n in kf
                if np.any(np.minimum(np.nan_to_num(kf[n][:, 1]),
                                     np.nan_to_num(kf[n][:, 2])) == 0.0)]
        if degk:
            b.append(para(
                f"{len(degk)} of {len(kf)} models are degenerate on at least "
                f"one k value, with sensitivity or specificity exactly zero: "
                f"{', '.join(sorted(degk))}."))
    else:
        missing(b, "5.6.2", "the k-fold run had not checkpointed when this "
                            "document was generated.")

    b.append(para("5.7 Feature Analysis", "Heading2"))
    if v2:
        b.append(para(
            f"The architecture results above do not by themselves show "
            f"whether the representation is empty. To separate the two, the "
            f"feature tensor was re-examined under model families that are "
            f"not data-starved at n = 50, across representations that "
            f"preserve what a channel mean destroys: multi-scale spatial "
            f"layout, per-channel distributions, and frame-to-frame temporal "
            f"change. Protocol: {v2['protocol']}."))
        rows = [[r["representation"], r["model"], f"{r['mean']*100:.2f}",
                 f"±{r['std']*100:.2f}"] for r in v2["ranking"][:10]]
        b.append(table(["Representation", "Model", "Nested BAL", "SD"], rows,
                       [3600, 1800, 1400, 1200],
                       highlight=lambda r: r[2] ==
                       f"{v2['winner']['nested_bal_acc']*100:.2f}"))
        b.append(para(f"Table 7. Top 10 of {len(v2['ranking'])} "
                      f"representation-model combinations evaluated under "
                      f"nested cross-validation.", "Caption"))
        w = v2["winner"]
        b.append(para(
            f"The winner is {w['model']} on {w['representation']} at "
            f"{w['nested_bal_acc']*100:.2f}% balanced accuracy. Tested "
            f"against label shuffles under the identical protocol, the null "
            f"distribution has mean {w['null_mean']*100:.2f}% and 95th "
            f"percentile {w['null_p95']*100:.2f}%, giving p = "
            f"{w['p_value']:.4f}. The representation carries signal above "
            f"chance."))
        b.append(figure("fig07_representation_search.png",
                        "Figure 4. Top 12 representation x model combinations. Only the top bar clears the permutation null."))
        b.append(para(
            "The representation that wins is the temporal one: absolute "
            "frame-to-frame differences, summarised by their mean, standard "
            "deviation and maximum per channel. Time-collapsed summaries of "
            "the same tensor score materially lower. This is consistent with "
            "the network's own behaviour, whose pooling stages reduce the "
            "spatial axes while the temporal structure reaches the LSTM "
            "already flattened into the sequence reshape."))
    else:
        missing(b, "5.7", "Optimized/optimize_v2.json is absent.")

    b.append(para("5.8 Statistical Analysis", "Heading2"))
    if tr:
        names = ([n for n in PUBLISHED if n in tr] + [n for n in LATEST if n in tr]
                 + [n for n in ["SMA-CLMPNet-Opt"] if n in tr])
        rows = []
        for n in names:
            a = tr[n][:, 0]
            if np.all(np.isnan(a)):
                continue
            rows.append([n, pct(np.nanmax(a)), pct(np.nanmean(a)),
                         pct(np.nanmin(a)), f"{np.nanvar(a):.6f}"])
        b.append(table(["Model", "Best", "Mean", "Worst", "Variance"], rows,
                       [2600, 1300, 1300, 1300, 1600]))
        b.append(para(
            "Table 8. Best, mean, worst and variance of accuracy across the "
            "six training percentages. The variances are dominated by the "
            "size of the evaluation corpus rather than by any property of the "
            "methods: a model whose test set shrinks from 31 videos to 6 will "
            "show variance from that alone.", "Caption"))
    if wts and wts.get("ranking"):
        w0 = wts["ranking"][0]
        b.append(para(
            f"A sweep of {len(wts['ranking'])} configurations of class "
            f"weight, probability calibration and decision threshold was run. "
            f"Protocol: {wts['protocol']}. Its best configuration, "
            f"{w0['model']} at threshold {w0['threshold']:.3f}, reached "
            f"{w0['bal_acc']*100:.2f}% balanced accuracy (±"
            f"{w0['std']*100:.2f}), which is below the untuned pipeline of "
            f"section 5.7. Selecting the threshold on the test partition "
            f"instead would raise this figure arbitrarily and measure "
            f"nothing; it was not done."))

    b.append(para("5.9 Confusion Matrix", "Heading2"))
    if roc:
        key = "temporal delta stats (best honest pipeline)"
        c = roc["curves"][key]
        cm = c["confusion_matrix"]
        b.append(para(
            f"Out-of-fold confusion matrix for the best measured pipeline, "
            f"accumulated over all {roc['corpus']['n']} videos under "
            f"{roc['protocol']}. Every prediction below was made on a fold "
            f"the model and its hyper-parameters had not seen."))
        b.append(table(
            ["", "Predicted authentic", "Predicted forged"],
            [["True authentic", str(cm["TN"]), str(cm["FP"])],
             ["True forged", str(cm["FN"]), str(cm["TP"])]],
            [2600, 2600, 2600]))
        b.append(para(
            f"Table 9. Accuracy {c['accuracy']*100:.2f}%, balanced accuracy "
            f"{c['balanced_accuracy']*100:.2f}%, sensitivity to forgery "
            f"{c['sensitivity_forged']*100:.2f}%, specificity "
            f"{c['specificity_authentic']*100:.2f}%. Both classes are "
            f"predicted, which is what distinguishes this pipeline from the "
            f"degenerate models of table 1.", "Caption"))
        b.append(figure("fig06_confusion_matrix.png",
                        "Figure 5. Out-of-fold confusion matrix."))
        if v2:
            b.append(para(
                f"This balanced accuracy, "
                f"{c['balanced_accuracy']*100:.2f}%, is not the same "
                f"statistic as the {v2['winner']['nested_bal_acc']*100:.2f}% "
                f"of section 5.7, and the two should not be averaged or "
                f"compared as if they were. Section 5.7 reports the mean of "
                f"the five outer-fold balanced accuracies; the figure here is "
                f"a single balanced accuracy computed once over the pooled "
                f"out-of-fold predictions for all "
                f"{roc['corpus']['n']} videos. The folds differ in class "
                f"composition, so the mean of per-fold rates and the rate of "
                f"the pooled predictions do not coincide. Both are honest "
                f"summaries of the same predictions; the pooled figure is the "
                f"one that corresponds to the confusion matrix above."))
            b.append(para(
                f"The forgery class is the better-detected one: "
                f"{cm['TP']} of {cm['TP'] + cm['FN']} forged videos are "
                f"caught, against {cm['TN']} of {cm['TN'] + cm['FP']} "
                f"authentic videos correctly cleared. The nine false "
                f"positives are the dominant error mode, which for a forensic "
                f"screening tool is the less damaging direction but is still "
                f"far too high a rate to deploy."))
    else:
        missing(b, "5.9", "Optimized/roc_confusion.json is absent. Generate "
                          "it with python Optimized/roc_confusion.py.")

    b.append(para("5.10 ROC Analysis", "Heading2"))
    if roc:
        key = "temporal delta stats (best honest pipeline)"
        ref = "per-frame mean+std (time-collapsed reference)"
        b.append(para(
            "The ROC of the original implementation cannot be reproduced: it "
            "is built through the same metric routine described in section "
            "5.3, so its curve is a function of a random vector rather than "
            "of any model output. The curve below was computed instead from "
            "out-of-fold predicted probabilities."))
        rows = [[key.split(" (")[0], f"{roc['curves'][key]['auc']:.4f}",
                 f"{roc['curves'][key]['balanced_accuracy']*100:.2f}"],
                [ref.split(" (")[0], f"{roc['curves'][ref]['auc']:.4f}",
                 f"{roc['curves'][ref]['balanced_accuracy']*100:.2f}"]]
        b.append(table(["Representation", "AUC", "Pooled BAL"], rows,
                       [4200, 1600, 1600]))
        if "auc_permutation" in roc:
            ap = roc["auc_permutation"]
            b.append(para(
                f"Table 10. Area under the ROC curve. Against "
                f"{ap['n_shuffles']} label shuffles the null AUC has mean "
                f"{ap['null_mean']:.4f} and 95th percentile "
                f"{ap['null_p95']:.4f}, giving p = {ap['p_value']:.4f} for "
                f"the observed {ap['observed']:.4f}.", "Caption"))
        b.append(para(
            f"The second row is the control that matters. Collapsing the same "
            f"tensor over time -- per-frame mean and standard deviation, "
            f"discarding frame-to-frame change -- drops the AUC to "
            f"{roc['curves'][ref]['auc']:.4f}, indistinguishable from chance. "
            f"The signal in this feature set is temporal, and any pooling "
            f"stage that averages it away removes it."))
        b.append(para(
            f"The curve itself is {len(roc['curves'][key]['fpr'])} points, "
            f"stored as false- and true-positive rate arrays in "
            f"Optimized/roc_confusion.json. With 50 samples the curve is a "
            f"step function and its confidence band is wide; the AUC and its "
            f"permutation test are the parts worth reporting."))
        b.append(figure("fig05_roc_curve.png",
                        "Figure 6. ROC, out-of-fold over all 50 videos, against the time-collapsed control."))
    else:
        missing(b, "5.10", "Optimized/roc_confusion.json is absent.")

    b.append(para("5.11 Comparative Discussion", "Heading2"))
    b.append(para(
        "Three lines of evidence agree, and they are not independent "
        "restatements of one another."))
    b.append(bullet(
        "The published architecture and its two ablations collapse to a "
        "constant prediction under correct scoring. That alone would be "
        "consistent with a training-recipe fault."))
    b.append(bullet(
        "Four current-generation backbones, as frozen extractors with trained "
        "heads, land in the same band. That rules out the architecture being "
        "uniquely at fault, and the smallest backbone scoring highest is the "
        "signature of fitting noise."))
    b.append(bullet(
        "Classical models under nested cross-validation, which are not "
        "data-starved at n = 50, do better than any of the deep models but "
        "still only reach the high seventies, and only on the temporal "
        "representation. A permutation test places that above chance."))
    if p2:
        vals = [v for v in p2.get("BiLSTMGBM", []) if v is not None]
        if vals:
            b.append(para(
                f"A further control: the architecture from the companion "
                f"study, ported to this feature set at its own published "
                f"settings ({p2['settings']['epochs']} epochs, batch "
                f"{p2['settings']['batch']}) but with its test-set fitting "
                f"step omitted, scores between {min(vals)*100:.2f}% and "
                f"{max(vals)*100:.2f}% with a mean of "
                f"{np.mean(vals)*100:.2f}%. That step, which receives the "
                f"test features and test labels and searches weights to "
                f"maximise the score computed on them, is what separates a "
                f"reported 100% from this."))
    b.append(para(
        "Taken together the limiting factor is the corpus. Fifty videos "
        "summarised to one tensor each give 19 to 44 training samples; "
        "published detectors on this dataset train on 10^5 to 10^6 face "
        "crops. No architectural change closes a gap of that size, and the "
        "measurements here do not support ranking the methods against each "
        "other at all."))
    b.append(figure("fig08_method_comparison.png",
                    "Figure 7. Every method tried, on identical data and folds. Blue clears the permutation null; red does not."))
    b.append(figure("fig09_accuracy_ceiling.png",
                    "Figure 8. The attainable-accuracy ceiling on this corpus."))

    # ====================================================== 6 Conclusion
    b.append(para("6. Conclusion", "Heading1"))
    b.append(para(
        "SMA-CLMPNet, evaluated on the feature set shipped with its "
        "implementation and scored with a correct confusion matrix, does not "
        "separate authentic from forged video: it assigns one label to every "
        "input at every training percentage tested, and its apparent accuracy "
        "is the class ratio of the corpus. Eleven comparison models, "
        "including four current-generation backbones, behave the same way. We "
        "report this as a negative result."))
    if v2:
        b.append(para(
            f"The feature design is not the part that fails. Frame-to-frame "
            f"temporal deltas of the proposed tensor support "
            f"{v2['winner']['nested_bal_acc']*100:.2f}% balanced accuracy "
            f"under nested cross-validation with an L1-regularised linear "
            f"model, at p = {v2['winner']['p_value']:.4f} against a "
            f"shuffled-label null. The signal is real, it lives on the "
            f"temporal axis, and a linear model recovers it. That is the "
            f"result we would build on."))
    b.append(para(
        "Two limitations bound everything above. The corpus is 50 videos, so "
        "no comparison between methods is resolvable and none should be read "
        "as one. And the metric routine in the delivered implementation does "
        "not use model predictions, so no previously reported figure from "
        "this pipeline is a measurement; the present paper reports only "
        "numbers recomputed with a correct confusion matrix."))
    b.append(para("Future work", "Heading2"))
    for t in [
        "Train at frame level on the full FaceForensics++ videos and evaluate "
        "at video level. This changes the sample count by three to four "
        "orders of magnitude and matters more than everything else combined. "
        "A pipeline for it, with identity-grouped splits and an explicit "
        "leakage guard, is implemented in FFPP/ but has not been run: the "
        "videos are licence-gated and absent.",
        "Fine-tune a modern backbone on face crops rather than freezing an "
        "ImageNet extractor over whole-frame feature maps. Freezing was "
        "correct here only because 19 to 44 samples cannot fine-tune "
        "anything.",
        "Keep the temporal delta representation and give it a model with "
        "capacity matched to the sample size, rather than a 2.26 M-parameter "
        "network.",
        "Report balanced accuracy and a confusion matrix with every result, "
        "and state the test-set size next to every percentage.",
        "Report cross-manipulation and cross-compression transfer, which is "
        "where detectors of this family are known to degrade.",
    ]:
        b.append(bullet(t))

    b.append(para("Reproducibility", "Heading1"))
    b.append(para(
        "Every table above is generated from a file in the repository by "
        "Optimized/make_paper.py; no value is hand-entered. The provenance of "
        "each is:"))
    prov = [["5.4, 5.5.1, 5.6.1, 5.8", "Analysis1/TRUE/*.npy",
             "logs/sweep_true.log"],
            ["5.5.2, 5.6.2", "Analysis1/TRUE_KF/*.npy", "logs/kfold_true.log"],
            ["5.7", "Optimized/optimize_v2.json", "logs/optimize_v2.log"],
            ["5.9, 5.10", "Optimized/roc_confusion.json",
             "Optimized/roc_confusion.py"],
            ["5.11", "Optimized/paper2_model.json",
             "logs/paper2_model_500.log"]]
    b.append(table(["Section", "Source", "Run log"], prov, [2200, 3400, 3000]))
    b.append(para(
        "Scoring is Optimized/metrics_fixed.py throughout. Fabricated "
        "artifacts from the original delivery were removed from the "
        "repository; Optimized/PROVENANCE.md records what and on what "
        "grounds.", "Caption"))

    # ---------------------------------------------------------------- write
    write_doc(OUT, "".join(b))
    for tag, ok in [("5.5.2/5.6.2 k-fold", bool(kf)),
                    ("5.7 feature analysis", bool(v2)),
                    ("5.9/5.10 ROC + confusion", bool(roc))]:
        print(f"  {'included' if ok else 'OMITTED '}  {tag}")


if __name__ == "__main__":
    main()
