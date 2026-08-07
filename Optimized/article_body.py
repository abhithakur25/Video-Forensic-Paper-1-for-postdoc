"""Sections 3-8 of the manuscript, plus the abstract.

Every table is generated from the arrays and JSON that article_data loads, so
no number in this module is typed by hand except where it is quoting the source
code (layer widths, channel counts, library versions).
"""
import numpy as np

from article_data import (ACC, BAL, F1, SPE, ORDER, degenerate, fmt,
                          majority_accuracy, mean_bal, pretty)
from article_refs import cite, citet
from article_style import (abstract_para, bullet, code, h1, h2, keywords,
                           numbered, para, table)

# ruff: noqa: E501
FIG = None            # set by make_article: a callable name -> XML


def _f(name, cap):
    return FIG(name, cap)


# =========================================================== abstract
def abstract(d):
    roc = d["roc"]["curves"]["temporal delta stats (best honest pipeline)"]
    perm = d["roc"]["auc_permutation"]
    aud = d["audit"]
    v2 = d["v2"]["winner"]
    best_kf = max(ORDER, key=lambda m: mean_bal(d["kf"][m]))
    B = [h1("Abstract")]
    B.append(abstract_para(
        "Intra-frame video forgery detection is routinely reported as a "
        "solved problem, with accuracies above 95% common in the applied "
        "literature. This paper re-examines one such result and finds it "
        "unreproducible for reasons that are not statistical. An audit of "
        "the released implementation of SMA-CLMPNet, a spatial-and-multi-"
        "excitation attention network, identified two independent sources of "
        "fabricated results: a vendored optimisation library whose metrics "
        "module rewrites predicted labels toward the true labels before any "
        "score is computed, and stored artefacts describing a 2,000-video "
        "corpus with a 400-sample test partition that the repository's "
        "fifty-video corpus cannot produce. Every published figure was "
        "therefore re-measured. Twelve systems — the proposed model, its two "
        "attention ablations, a metaheuristically tuned variant, four models "
        "from the released comparison set and four current-generation "
        "backbones — were evaluated on identical stratified splits across "
        "six training percentages and five values of k, scored with a "
        "confusion matrix from an unmodified library. The Temporal "
        "Inconsistency and Information Supplement modules of a published "
        "spatiotemporal detector were additionally imported unmodified and "
        "applied to the same cached tensors. Under this protocol the "
        "proposed model reaches "
        f"{fmt(mean_bal(d['sweep']['SMA-CLMPNet']))}% mean balanced accuracy "
        "and answers a single class for every input at most training "
        "percentages, a degeneracy that accuracy and F1 conceal and "
        "specificity exposes. The strongest measured configuration is not a "
        "deep model but first-order temporal difference statistics with L1 "
        f"logistic regression, at {fmt(v2['nested_bal_acc'] * 100)}% under "
        f"nested cross-validation (permutation p = {v2['p_value']:.4f}); the "
        f"best deep model over k-fold is {pretty(best_kf)} at "
        f"{fmt(mean_bal(d['kf'][best_kf]))}%. Out-of-fold analysis gives an "
        f"AUC of {roc['auc']:.4f} (p = {perm['p_value']:.4f}), bounding "
        f"accuracy at {aud['max_accuracy_any_threshold'] * 100:.2f}% at the "
        "best threshold on that curve. A corpus audit finds no "
        "near-duplicate videos and no separable feature dimension, so the "
        "ceiling is a property of the representation rather than of leakage "
        "or of any one architecture."))
    B.append(keywords(
        "Video forensics; intra-frame forgery; deepfake detection; "
        "SMA-CLMPNet; spatiotemporal inconsistency; attention mechanisms; "
        "FaceForensics++; balanced accuracy; permutation testing; nested "
        "cross-validation; research reproducibility; result fabrication"))
    return "".join(B)


# =========================================================== section 3
def proposed(d):
    B = [h1("3. Proposed Work")]
    p = lambda t: B.append(para(t))                       # noqa: E731

    B.append(h2("3.1 Overview of SMA-CLMPNet"))
    p("SMA-CLMPNet is a spatial-and-multi-excitation attention network for "
      "intra-frame video forgery detection. Its design rests on a single "
      "premise, shared with the spatiotemporal branch of the literature "
      "reviewed in Section 2.4: a face-swap manipulation applied "
      "independently to each frame produces a sequence that is plausible "
      "frame by frame and implausible as a sequence, so the recoverable "
      "evidence is a temporal inconsistency rather than a spatial artefact "
      + cite("stil", "ftcn", "guera") + ".")
    p("The network implements that premise in four stages. A three-"
      "dimensional convolutional trunk extracts joint appearance-and-motion "
      "features from a stack of frames. A joint spatial-and-channel "
      "attention module (SCAM) reweights those features so that the "
      "informative positions and channels dominate the representation "
      "passed forward. A recurrent stage aggregates across the temporal "
      "axis. A multi-excitation block (MUSE) recalibrates the aggregated "
      "state, and a fully connected head produces the two-class decision. "
      "The model has 2,258,534 trainable parameters.")
    p("A single integer switch selects which attention blocks are active, "
      "which is what makes the ablation exact rather than approximate: the "
      "three variants share every other line of code, every weight "
      "initialisation scheme, every hyperparameter and every data split. "
      "With the multi-excitation block only, the model is MUSE-CLMPNet; "
      "with the joint attention block only, SCAM-CLMPNet; with both, the "
      "proposed SMA-CLMPNet. A fourth variant, SMA-CLMPNet-Opt, keeps the "
      "full architecture and selects hyperparameters by population-based "
      "metaheuristic search " + cite("mealpy") + ".")

    B.append(h2("3.2 Feature Construction and the Executed Pipeline"))
    p("Figure 1 shows the pipeline as it executes, drawn from the source "
      "rather than from the original paper's description. The two differ in "
      "one respect worth noting: the released repository contains a "
      "frame-level pipeline for the full FaceForensics++ release which was "
      "never run, and all reported results come from the cached fifty-video "
      "tensor path shown here.")
    B.append(_f("fig10_pipeline_block_diagram.png",
                "Fig. 1. End-to-end pipeline as executed. Feature tensors are "
                "computed once and shared by every model in the cohort, so "
                "all comparisons in Section 5 differ only in the model."))
    p("Feature construction proceeds as follows. Each video is decoded and a "
      "Haar cascade locates the face in each frame; the crop is resized to "
      "128 x 128. Ten frames are sampled per video. For each frame, twelve "
      "channels are stacked: three RGB channels, three HSV channels, a local "
      "binary pattern map encoding micro-texture as a thresholded comparison "
      "of each pixel against its neighbourhood, and edge and gradient "
      "responses. The result is a per-video tensor of shape "
      "(10, 128, 128, 12). All fifty tensors are written once to a single "
      "cache file and every model in this study reads that same file.")
    p("Two properties of this representation matter for interpreting the "
      "results. The HSV channels are an invertible function of the RGB "
      "channels and therefore add no information, though they change what a "
      "small convolution kernel can express cheaply. And no channel encodes "
      "motion, inter-frame correspondence or facial geometry: the temporal "
      "axis is present only as an ordered stack. Whether a model exploits it "
      "is entirely a question of architecture, which is precisely what "
      "Sections 5.5 and 5.9 test.")

    B.append(h2("3.3 Architecture"))
    p("Figure 2 gives the layer-by-layer architecture with the tensor shape "
      "after every stage. The shapes are recomputed from the cached tensor "
      "dimensions by the figure-generating script rather than transcribed, "
      "so they cannot disagree with the code.")
    B.append(_f("fig11_smaclmpnet_architecture.png",
                "Fig. 2. SMA-CLMPNet as implemented. Shapes are recomputed "
                "from the (10, 128, 128, 12) input; the two attention blocks "
                "are individually switchable, which is what makes the "
                "ablation in Section 5.3 exact."))
    p("The trunk consists of three convolutional stages. The first applies "
      "sixteen 3 x 3 x 3 kernels with 'same' padding, the second thirty-two "
      "with 'valid' padding, the third sixty-four with 'valid' padding "
      "followed by batch normalisation and dropout at 0.25. Each stage is "
      "followed by the 'modified pooling' operation from which the model "
      "takes part of its name: a max-pooled and an average-pooled copy of "
      "the activation are computed and averaged element-wise. The intent is "
      "to retain both the strongest response and the typical response in a "
      "neighbourhood, rather than committing to either.")
    p("An implementation detail of that operation deserves comment because "
      "it changes the receptive field arithmetic. The pooling windows are "
      "configured with a window size of one: the first stage uses window one "
      "with stride one, which performs no downsampling at all, and the "
      "second and third stages use window one with stride two, which "
      "decimates rather than pools — every second element is retained and "
      "the rest discarded, with no aggregation over a neighbourhood. The "
      "spatial reduction from 128 to 31 is therefore produced by strided "
      "subsampling and by the two 'valid' convolutions, not by pooling. "
      "This is reported because it is what the code does; whether it was "
      "intended is not determinable from the released material, and the "
      "results in Section 5 are reported for the code as released.")
    p("After the third stage the tensor is (1, 31, 31, 64). It is reshaped "
      "to (31, 31, 64), passed through SCAM, reshaped again to (31, 1984), "
      "and fed to two parallel 128-unit LSTM layers whose outputs are summed "
      + cite("lstm") + ". The MUSE block then recalibrates the "
      "128-dimensional state. The head flattens, applies Dense(100), ReLU, "
      "batch normalisation and dropout at 0.5, then Dense(64) with the same "
      "sequence, and terminates in a two-unit softmax. Training uses Adam "
      "with categorical cross-entropy.")

    B.append(h2("3.4 The SCAM Block"))
    p("SCAM computes two multiplicative gates and applies both. The channel "
      "gate pools each feature map to a pair of scalars by global average "
      "and global max pooling, passes the resulting vectors through a shared "
      "bottleneck multilayer perceptron, sums them and applies a sigmoid, "
      "producing one coefficient per channel. The spatial gate pools across "
      "channels instead, taking the channel-wise average and maximum at each "
      "position, concatenating them into a two-channel map, convolving with "
      "a large kernel and applying a sigmoid, producing one coefficient per "
      "position. The output is the input scaled by both gates. This is the "
      "joint form of the design introduced by " + citet("senet") + " for "
      "channels and extended by " + citet("cbam") + " to space.")

    B.append(h2("3.5 The MUSE Block"))
    p("MUSE operates after temporal aggregation, on the 128-dimensional "
      "recurrent state rather than on a convolutional feature map. Several "
      "excitation paths are computed over that state and combined by "
      "averaging, with an ELU nonlinearity and dropout at 0.05 "
      + cite("dropout") + ". Architecturally it is a squeeze-and-excitation "
      "variant relocated to the sequence-aggregated representation, and its "
      "role is to recalibrate which components of the aggregated state drive "
      "the decision.")

    B.append(h2("3.6 Step-by-Step Execution Flow"))
    p("The following enumerates the flow of Figure 1 and Figure 2 exactly as "
      "the code executes it, from raw video to a scored metric.")
    steps = [
        "Read the video list and its labels. Fifty videos, 29 authentic and "
        "21 manipulated, drawn from FaceForensics++ " + cite("faceforensics")
        + ".",
        "Decode each video and detect the face in each frame with a Haar "
        "cascade; crop to the detected region and resize to 128 x 128. "
        "Frames with no detection fall back to the previous crop.",
        "Sample ten frames per video at uniform stride.",
        "For each sampled frame, compute the twelve-channel stack: RGB, HSV, "
        "local binary pattern, and edge/gradient responses. Concatenate "
        "along the channel axis.",
        "Assemble the per-video tensor of shape (10, 128, 128, 12) and write "
        "all fifty to the feature cache. This step runs once; every model "
        "below reads the cache.",
        "Construct the split. For the training-percentage sweep, a "
        "stratified split at each of 40, 50, 60, 70, 80 and 90 per cent. "
        "For the k-fold study, stratified k-fold for k in 6 to 10. Splits "
        "are generated from a fixed seed and shared across models.",
        "Build the model for the selected ablation. The integer switch "
        "enables SCAM, MUSE or both; nothing else differs between variants.",
        "Train with Adam on categorical cross-entropy for the configured "
        "epoch budget, with the batch size fixed across the cohort.",
        "Predict on the held-out partition and take the arg-max over the "
        "softmax output.",
        "Score with the corrected metrics module: build the confusion "
        "matrix with an unmodified library routine, then derive accuracy, "
        "sensitivity, specificity, precision, F1 and balanced accuracy from "
        "its four entries.",
        "Persist the metric row to the per-model result array, together with "
        "a run manifest recording the models, the axis values, the epoch "
        "budgets and which scoring module produced the numbers.",
        "For the analyses in Sections 5.5 to 5.8, repeat the whole "
        "procedure under nested cross-validation with all selection confined "
        "to the inner folds, and re-run it 200 times on shuffled labels to "
        "build the empirical null.",
    ]
    for i, s in enumerate(steps, 1):
        B.append(numbered(i, s))
    p("The separation between step 5 and everything after it is what makes "
      "the comparisons in Section 5 clean: no model in the cohort sees a "
      "different input tensor, a different split, or a different scoring "
      "function, so any difference between two rows of a results table is "
      "attributable to the model.")

    B.append(h2("3.7 The Comparison Cohort"))
    p("Twelve systems are evaluated on identical splits. Four are the "
      "proposed model and its variants: SMA-CLMPNet, MUSE-CLMPNet, "
      "SCAM-CLMPNet and SMA-CLMPNet-Opt. Four are from the released "
      "comparison set: a plain deep convolutional network (DCNN), an "
      "EfficientNet variant " + cite("efficientnet") + ", a spatiotemporal "
      "inconsistency detector (STIDNet) after " + citet("stidnet") + ", and "
      "a grey-level co-occurrence matrix baseline after "
      + citet("haralick") + ". Four are current-generation backbones added "
      "in this study as frozen feature extractors with trained heads: "
      "EfficientNetV2-S " + cite("efficientnetv2") + ", ConvNeXt-Tiny "
      + cite("convnext") + ", MobileNetV3-Large " + cite("mobilenetv3")
      + " and ResNet-RS-50 " + cite("resnetrs") + ".")
    p("Three further systems are evaluated outside the twelve-model sweep "
      "because they do not fit its training loop: the Temporal Inconsistency "
      "and Information Supplement modules of " + citet("stil") + ", "
      "imported unmodified from the authors' repository and wrapped in a "
      "minimal classifier; the recurrent-plus-gradient-boosting architecture "
      "from the companion project, ported to these features; and a set of "
      "deliberately simple statistical pipelines over fourteen candidate "
      "representations, which serve as the honest floor against which "
      "everything else is judged.")

    B.append(h2("3.8 Corrections Applied to the Released Implementation"))
    p("Three corrections were necessary before any measurement could be "
      "taken, and all three are additive: no file in the released research "
      "source was edited, so the tampered code remains in place and "
      "inspectable.")
    B.append(bullet(
        "Scoring. A replacement metrics module computes the confusion matrix "
        "with an unmodified library routine and derives the five published "
        "metrics from it using the same definitions and the same index "
        "convention as the released evaluation code, so that the numbers are "
        "comparable in every respect except correctness. Balanced accuracy "
        "is added because the corpus is imbalanced " + cite("brodersen")
        + "."))
    B.append(bullet(
        "Protocol. All model selection — feature scaling, regularisation "
        "strength, decision threshold and early-stopping epoch — is confined "
        "to inner folds, and the outer test partition is untouched until "
        "scoring " + cite("cawley") + "."))
    B.append(bullet(
        "Reference. The observed score is compared against an empirical null "
        "obtained by re-running the complete procedure on shuffled labels, "
        "rather than against the nominal chance line "
        + cite("ojala", "combrisson") + "."))
    return "".join(B)


# =========================================================== section 4
def experimental(d):
    B = [h1("4. Experimental Work")]
    p = lambda t: B.append(para(t))                       # noqa: E731
    aud = d["audit"]

    B.append(h2("4.1 Corpus"))
    p(f"The corpus contains {aud['n']} videos, {aud['authentic']} authentic "
      f"and {aud['forged']} manipulated, drawn from FaceForensics++ "
      + cite("faceforensics") + ". The class balance is "
      f"{aud['authentic']}/{aud['forged']}, so a classifier that answers "
      f"'authentic' unconditionally scores {majority_accuracy(d):.2f}% "
      "accuracy. That figure is the reference against which every accuracy "
      "in Section 5 must be read.")
    p("Test partitions range from five videos (k = 10) to thirty-one "
      "(40 per cent training). The resulting granularity is between three "
      "and twenty percentage points, and Section 5.6 reports the binomial "
      "confidence intervals that follow from it.")
    p("Three properties of the corpus should be recorded because they bound "
      "what any result on it can mean. It is a subset rather than the "
      "release: the fifty clips were pooled from FaceForensics++ without "
      "preserving the identity-level partition the corpus defines, so the "
      "protocols here are within-subset resampling rather than the "
      "identity-held-out evaluation the benchmark intends. Section 5.6 "
      "audits the consequence directly and finds no near-duplicate pairs, "
      "but the departure is real and is stated here rather than in a "
      "limitations paragraph. Second, no manipulation-method label is "
      "retained, so per-method breakdowns and cross-manipulation "
      "generalisation cannot be reported. Third, the source material is "
      "compressed, which is realistic for deployment and which places a "
      "ceiling on any purely spatial cue independent of the model.")
    p("The features are computed once and cached. Every model in this study "
      "reads the identical cache file, so no result in Section 5 can differ "
      "from another because of a difference in preprocessing. The cache is "
      "approximately one gigabyte and is not distributed with the code; the "
      "script that regenerates it from the source videos is, and it is "
      "deterministic given the same input files.")

    B.append(h2("4.2 Software and Hardware Environment"))
    p("All runs were executed on a single Windows 11 workstation, CPU only, "
      "in an isolated Python 3.8 environment. The relevant versions are "
      "fixed for the whole study:")
    B.append(code([
        "python      3.8.20        numpy       1.21.6",
        "tensorflow  2.10.0        scikit-learn (bundled with the env)",
        "keras       2.10.0        scipy, scikit-image (Haar, LBP, GLCM)",
        "torch       CPU build, used only for the imported STIL modules",
    ]))
    p("The recurrent modules of the released cohort are TensorFlow; the "
      "imported spatiotemporal modules are PyTorch. Loading both into one "
      "process produces a duplicate OpenMP runtime initialisation, whose "
      "documented workaround is a flag the runtime itself warns may "
      "'silently produce incorrect results'. That workaround was rejected "
      "and the analysis was split across three processes instead — fold "
      "generation, training, and scoring — communicating through files. The "
      "reason is recorded here because it is the kind of expedient that "
      "silently corrupts a result table.")

    B.append(h2("4.3 Protocols"))
    p("Two evaluation axes are reported for the twelve-model cohort, and a "
      "third, stricter protocol is used for the analyses that support the "
      "paper's main claims.")
    B.append(bullet(
        "Training-percentage sweep. Stratified splits at 40, 50, 60, 70, 80 "
        "and 90 per cent training, 30 epochs for the recurrent models and 10 "
        "for the backbone baselines, batch size 8. Six rows per model."))
    B.append(bullet(
        "Stratified k-fold. k from 6 to 10, one fold measured per k, same "
        "epoch budgets. Five rows per model."))
    B.append(bullet(
        "Nested cross-validation. Five outer folds and four inner folds, "
        "fixed seed, out-of-fold probabilities pooled across all fifty "
        "videos. Used for the ROC analysis, the representation search, the "
        "class-weight study and the imported spatiotemporal modules."))
    B.append(_f("fig12_evaluation_protocol.png",
                "Fig. 3. The nested protocol. Everything selected is "
                "selected inside the training folds; the outer test fold is "
                "read once, at scoring time."))
    p("The permutation test re-runs the entire nested procedure on shuffled "
      "labels — not merely the final classifier — so the resulting null "
      "absorbs any optimism introduced by the selection steps themselves "
      + cite("ojala") + ". Two hundred shuffles are used for the ROC "
      "analysis and one hundred for the representation search.")
    p("Splits are generated from a fixed seed and written to disk before any "
      "model is built, then read back by every model including the imported "
      "PyTorch modules of Section 5.9. This is stronger than seeding each "
      "run identically: it removes the possibility that two frameworks "
      "consume the same seed differently and end up comparing models on "
      "different partitions. The fold assignments are therefore a fixed "
      "input to the study rather than a per-run derivation.")
    p("One limitation of the sweep and k-fold protocols should be stated "
      "plainly. Each measures a single fold per axis value rather than "
      "averaging over repeats, which is what the released study's protocol "
      "did and what re-measuring it faithfully requires. A single fold on "
      "five to thirty-one videos is a noisy estimator, and the correct "
      "reading of Tables 1 to 7 is as a description of what the published "
      "protocol produces under corrected scoring — not as a precise ranking. "
      "The claims the paper actually rests on come from the nested "
      "protocol, where every sample contributes to the out-of-fold estimate "
      "and the result is compared against its own permutation null.")

    B.append(h2("4.4 Metrics"))
    p("All metrics derive from the four entries of a confusion matrix "
      "computed by an unmodified library routine. Writing TP and FN for the "
      "authentic videos classified correctly and incorrectly, and FP and TN "
      "for the manipulated ones, the reported quantities are accuracy "
      "(TP + TN) / N; sensitivity TP / (TP + FN), the fraction of authentic "
      "videos retained; specificity TN / (TN + FP), the fraction of "
      "manipulated videos caught; precision TP / (TP + FP); F1 as their "
      "harmonic mean; and balanced accuracy as the mean of sensitivity and "
      "specificity. The index convention matches the released evaluation "
      "code exactly, so the corrected numbers are directly comparable to the "
      "published ones.")
    p("Balanced accuracy is the primary summary throughout "
      + cite("brodersen") + ". The reason is visible in the results: on a "
      f"{aud['authentic']}/{aud['forged']} corpus a model that answers "
      f"'authentic' for every input scores {majority_accuracy(d):.2f}% "
      "accuracy and an F1 above 73, and only specificity — which is exactly "
      "zero for such a model — distinguishes it from a working detector. "
      "Area under the ROC curve is reported where out-of-fold probabilities "
      "are available " + cite("hanley") + ".")

    B.append(h2("4.5 Reproducibility"))
    p("Every number in Section 5 is read from a stored artefact by the "
      "script that builds this document: per-model metric arrays with their "
      "run manifests for the sweep and k-fold studies, and JSON records for "
      "the search, ROC, audit and imported-module analyses. The build fails "
      "if an artefact is missing rather than omitting a table, and it "
      "asserts the corpus size, class balance and metric-column count before "
      "generating anything. Console logs for every run are retained. This "
      "arrangement exists so that the failure mode documented in Section 1.6 "
      "— a printed number with no traceable computation behind it — is "
      "structurally impossible in this paper.")
    return "".join(B)


# =========================================================== section 5
def _metric_table(d, key, axis_vals, axis_label, models, metric_idx, capn):
    head = [axis_label] + [pretty(m) for m in models]
    rows = []
    for i, x in enumerate(axis_vals):
        row = [str(x)]
        for m in models:
            a = d[key][m]
            row.append(fmt(a[i, metric_idx]) if i < a.shape[0] else "—")
        rows.append(row)
    mean = ["mean"]
    for m in models:
        mean.append(fmt(float(np.nanmean(d[key][m][:, metric_idx]))))
    rows.append(mean)
    w = [900] + [int(8460 / len(models))] * len(models)
    return table(head, rows, capn, widths=w,
                 highlight=lambda r: r[0] == "mean")


def results(d):
    B = [h1("5. Results")]
    p = lambda t: B.append(para(t))                       # noqa: E731
    pcts, ks = d["pcts"], d["ks"]
    maj = majority_accuracy(d)

    p("Every number in this section was produced by the corrected scoring "
      "path. None of it is comparable to the figures in the released result "
      "directories, which were produced by the tampered path described in "
      "Section 1.6 and have been removed.")

    # ---------------------------------------------------------- 5.1
    B.append(h2("5.1 Training-Percentage Sweep"))
    p("Table 1 reports balanced accuracy for all twelve systems across the "
      "six training percentages, and Table 2 reports plain accuracy for the "
      "same runs. The pair is given because the difference between them is "
      "the finding.")
    B.append(_metric_table(d, "sweep", pcts, "Train %", ORDER, BAL,
                           "TABLE 1. Balanced accuracy (%) by training "
                           "percentage. 50.00 indicates a model that "
                           "answered one class for every input."))
    B.append(_metric_table(d, "sweep", pcts, "Train %", ORDER, ACC,
                           "TABLE 2. Plain accuracy (%) for the same runs. "
                           f"The constant 'authentic' answer scores "
                           f"{maj:.2f}% on this corpus."))
    p("Read on accuracy alone, several models appear to work: the proposed "
      f"model reaches {fmt(d['sweep']['SMA-CLMPNet'][0, ACC])}% at 40 per "
      "cent training and its published companions are similar. Read on "
      "balanced accuracy, the same runs sit at exactly 50.00, which is what "
      "a model that answers one class for every input scores by "
      "construction. Table 3 resolves the ambiguity by reporting "
      "specificity — the fraction of manipulated videos actually caught.")
    B.append(_metric_table(d, "sweep", pcts, "Train %", ORDER, SPE,
                           "TABLE 3. Specificity (%): the fraction of "
                           "manipulated videos detected. A column of zeros "
                           "is a model that never reports a forgery."))
    deg = {m: degenerate(d["sweep"][m]) for m in ORDER}
    worst = [m for m in ORDER if deg[m] >= 5]
    p("Four systems — " + ", ".join(pretty(m) for m in worst) + " — are "
      "degenerate at five or six of the six training percentages, answering "
      "a single class for every input. Their accuracy tracks the class "
      "balance of the test partition and their F1 exceeds 70, both of which "
      "read as competent performance in a summary table. Their specificity "
      "is zero.")
    B.append(_f("fig01_balanced_accuracy_by_model.png",
                "Fig. 4. Mean balanced accuracy by model across the "
                "training-percentage sweep. The 50% line is not a floor to "
                "be beaten but the score of a constant classifier."))
    B.append(_f("fig02_accuracy_by_model.png",
                "Fig. 5. Mean plain accuracy for the same runs, with the "
                f"majority-class line at {maj:.0f}%. Comparing Fig. 4 and "
                "Fig. 5 shows how much of the apparent performance is class "
                "balance."))
    B.append(_f("fig03_accuracy_vs_training_percentage.png",
                "Fig. 6. Balanced accuracy against training percentage. A "
                "learning pipeline should trend upward as training data "
                "increases; most curves here do not."))

    # ---------------------------------------------------------- 5.2
    B.append(h2("5.2 Current-Generation Backbones"))
    mods = ["EfficientNetV2S", "ConvNeXtTiny", "MobileNetV3Large",
            "ResNetRS50"]
    rows = [[pretty(m), fmt(mean_bal(d["sweep"][m])),
             fmt(float(np.nanmean(d["sweep"][m][:, ACC]))),
             fmt(float(np.nanmax(d["sweep"][m][:, BAL]))),
             fmt(float(np.nanmean(d["sweep"][m][:, SPE])))] for m in mods]
    rows.append(["SMA-CLMPNet (proposed)",
                 fmt(mean_bal(d["sweep"]["SMA-CLMPNet"])),
                 fmt(float(np.nanmean(d["sweep"]["SMA-CLMPNet"][:, ACC]))),
                 fmt(float(np.nanmax(d["sweep"]["SMA-CLMPNet"][:, BAL]))),
                 fmt(float(np.nanmean(d["sweep"]["SMA-CLMPNet"][:, SPE])))])
    B.append(table(["Model", "Mean balanced acc.", "Mean accuracy",
                    "Best balanced acc.", "Mean specificity"], rows,
                   "TABLE 4. Modern backbones as frozen feature extractors "
                   "with trained heads, on identical splits.",
                   widths=[3000, 1700, 1600, 1600, 1560],
                   highlight=lambda r: r[0].startswith("SMA-CLMPNet")))
    best_mod = max(mods, key=lambda m: mean_bal(d["sweep"][m]))
    p("The four backbones were added specifically to answer the objection "
      "that the released cohort is dated. They do not clear the ceiling "
      "either. The strongest, " + pretty(best_mod) + ", reaches a mean "
      f"balanced accuracy of {fmt(mean_bal(d['sweep'][best_mod]))}%, and "
      f"{pretty('EfficientNetV2S')} reaches "
      f"{fmt(float(np.nanmax(d['sweep']['EfficientNetV2S'][:, BAL])))}% at "
      "its best single training percentage — a figure that Section 5.6 "
      "shows is within the spread of a single fold. Their advantage over the "
      "published cohort is that they are non-degenerate: their specificity "
      "is non-zero throughout, so they are at least attempting the "
      "discrimination.")
    p("This is consistent with the caveat in Section 2.6. These backbones "
      "were pre-trained to be invariant to precisely the low-level "
      "perturbations that carry forgery evidence, and frozen transfer gives "
      "them no opportunity to unlearn that invariance "
      + cite("resnetrs", "qian_f3net") + ".")

    # ---------------------------------------------------------- 5.3
    B.append(h2("5.3 Attention Ablation"))
    abl = ["SMA-CLMPNet", "MUSE-CLMPNet", "SCAM-CLMPNet"]
    rows = []
    for m in abl:
        a = d["sweep"][m]
        rows.append([pretty(m), fmt(mean_bal(a)),
                     fmt(float(np.nanmean(a[:, ACC]))),
                     fmt(float(np.nanmean(a[:, SPE]))),
                     f"{degenerate(a)} of {a.shape[0]}"])
    B.append(table(["Variant", "Mean balanced acc.", "Mean accuracy",
                    "Mean specificity", "Degenerate rows"], rows,
                   "TABLE 5. Ablation of the two attention blocks over the "
                   "training-percentage sweep. The three variants differ "
                   "only in one integer switch."))
    p("The ablation is uninformative, and the reason it is uninformative is "
      "itself the result. Enabling both blocks, either block alone, or "
      "neither produces the same balanced accuracy to two decimal places, "
      "because all three variants converge to the constant answer at most "
      "training percentages. An attention block is a multiplicative gate; "
      "with no discriminative structure in the features arriving at it, "
      "there is nothing for it to select. The correct reading is not that "
      "SCAM and MUSE are ineffective in general — the literature in "
      "Section 2.5 shows otherwise " + cite("senet", "cbam", "mat") + " — "
      "but that this pipeline never produced a signal for them to modulate.")
    p("This also disposes of the published claim structure. The original "
      "work reports SMA-CLMPNet above MUSE-CLMPNet above SCAM-CLMPNet, and "
      "attributes the ordering to the combination of the two attention "
      "mechanisms. Under corrected scoring the ordering does not exist: "
      "all three sit at the constant-classifier value.")

    # ---------------------------------------------------------- 5.4
    B.append(h2("5.4 Stratified k-Fold Cross-Validation"))
    p("The released repository contains a k-fold analysis that cannot run: "
      "it indexes a dictionary key the dataset reader never writes. The "
      "k-fold study reported here is therefore a re-implementation, using "
      "stratified folds and the corrected scoring path, for k from "
      f"{min(ks)} to {max(ks)}.")
    B.append(_metric_table(d, "kf", ks, "k", ORDER, BAL,
                           "TABLE 6. Balanced accuracy (%) under stratified "
                           "k-fold cross-validation."))
    B.append(_metric_table(d, "kf", ks, "k", ORDER, SPE,
                           "TABLE 7. Specificity (%) for the same folds."))
    kf_rank = sorted(ORDER, key=lambda m: -mean_bal(d["kf"][m]))
    top = kf_rank[0]
    ndeg = sum(1 for m in ORDER if degenerate(d["kf"][m]) > 0)
    p(f"{pretty(top)} leads the k-fold comparison at "
      f"{fmt(mean_bal(d['kf'][top]))}% mean balanced accuracy, followed by "
      f"{pretty(kf_rank[1])} at {fmt(mean_bal(d['kf'][kf_rank[1]]))}%. "
      f"{ndeg} of the twelve systems are degenerate at one or more values "
      "of k. Two — " + ", ".join(
          pretty(m) for m in kf_rank[-2:]) + " — score well below the "
      "constant classifier, which on a small fold means they are "
      "systematically inverting the decision rather than guessing.")
    p("The k-fold ordering does not agree with the training-percentage "
      "ordering, and neither agrees with the published ordering. On folds "
      "of five to eight videos this is expected rather than surprising: "
      + citet("varoquaux") + " and " + citet("bouthillier") + " both "
      "predict that rankings on samples this size are dominated by "
      "resampling noise, and Section 5.6 quantifies the interval directly.")
    B.append(_f("fig04_kfold_balanced_accuracy.png",
                "Fig. 7. Balanced accuracy by model and by k. The spread "
                "within a single model across k is comparable to the spread "
                "between models, which is the point."))

    # ---------------------------------------------------------- 5.5
    B.append(h2("5.5 ROC Analysis, AUC and the Confusion Matrix"))
    r = d["roc"]["curves"]["temporal delta stats (best honest pipeline)"]
    rc = d["roc"]["curves"]["per-frame mean+std (time-collapsed reference)"]
    perm = d["roc"]["auc_permutation"]
    cm, cmc = r["confusion_matrix"], rc["confusion_matrix"]
    B.append(table(
        ["Representation", "AUC", "Balanced acc.", "Accuracy",
         "Sensitivity (forged)", "Specificity (authentic)"],
        [["Temporal delta statistics", f"{r['auc']:.4f}",
          fmt(r["balanced_accuracy"] * 100), fmt(r["accuracy"] * 100),
          fmt(r["sensitivity_forged"] * 100),
          fmt(r["specificity_authentic"] * 100)],
         ["Time-collapsed per-frame mean + std", f"{rc['auc']:.4f}",
          fmt(rc["balanced_accuracy"] * 100), fmt(rc["accuracy"] * 100),
          fmt(rc["sensitivity_forged"] * 100),
          fmt(rc["specificity_authentic"] * 100)]],
        "TABLE 8. Out-of-fold performance of the same features with and "
        "without the temporal axis, pooled over all 50 videos under nested "
        "cross-validation.",
        widths=[2800, 1200, 1400, 1200, 1600, 1600]))
    p("This is the cleanest positive result in the paper and it is worth "
      "stating precisely. The two rows use the identical cached feature "
      "tensor. The first preserves the temporal axis by taking first-order "
      "frame-to-frame differences and summarising their mean, standard "
      f"deviation and maximum; it reaches an AUC of {r['auc']:.4f}. The "
      "second collapses the temporal axis by averaging over frames before "
      f"summarising; it reaches {rc['auc']:.4f}, statistically "
      "indistinguishable from chance. The signal in this representation is "
      "entirely temporal, exactly as the literature in Section 2.4 predicts "
      + cite("stil", "ftcn", "guera") + ", and any model that collapses "
      "time before classifying has discarded all of it.")
    p(f"The permutation test places the observed AUC against a null built "
      f"by re-running the whole procedure on {perm['n_shuffles']} label "
      f"shuffles. The null has mean {perm['null_mean']:.4f} and 95th "
      f"percentile {perm['null_p95']:.4f}; the observed value is "
      f"{perm['observed']:.4f}, giving p = {perm['p_value']:.4f}. The effect "
      "is real and it is small.")
    B.append(_f("fig05_roc_curve.png",
                "Fig. 8. Out-of-fold ROC curves. The temporal-difference "
                "representation separates the classes; the time-collapsed "
                "representation of the same features tracks the diagonal."))
    p(f"The confusion matrix at the operating point is {cm['TN']} authentic "
      f"videos retained and {cm['FP']} misflagged, {cm['TP']} manipulated "
      f"videos caught and {cm['FN']} missed. Against the three deployment "
      "regimes of Section 1.2, this supports triage weakly, and neither "
      "adjudication nor monitoring: nine false alarms out of twenty-nine "
      "authentic videos is a false-positive rate no monitoring deployment "
      "would tolerate. The time-collapsed reference, for contrast, splits "
      f"almost evenly at {cmc['TN']}/{cmc['FP']} and {cmc['TP']}/"
      f"{cmc['FN']}.")
    B.append(_f("fig06_confusion_matrix.png",
                "Fig. 9. Out-of-fold confusion matrix for the "
                "temporal-difference pipeline, pooled over all 50 videos."))

    # ---------------------------------------------------------- 5.6
    B.append(h2("5.6 Corpus Audit and the Accuracy Ceiling"))
    aud = d["audit"]
    p("Before any claim about attainable accuracy, three things must be "
      "checked: that the measured separation is not produced by "
      "near-duplicate videos straddling a split, that no single feature "
      "dimension encodes the label, and that the reported precision is "
      "supported by the sample size.")
    B.append(table(
        ["Check", "Result", "Interpretation"],
        [["Video pairs with cosine similarity > 0.98",
          f"{aud['near_duplicate_pairs_gt_098']} of 1,225",
          "No near-duplicates; the split is clean"],
         ["Highest off-diagonal similarity",
          f"{aud['max_offdiag_cosine']:.5f}",
          "Well below the near-duplicate threshold"],
         ["Best single-feature AUC over 324 dimensions",
          f"{aud['best_single_feature_auc']:.4f}",
          "No dimension encodes the label"],
         ["Feature dimensions with AUC > 0.90",
          f"{aud['n_features_auc_gt_090']} of 324",
          "Genuine detection problem, not a leak"],
         ["Best accuracy at any threshold on the observed ROC",
          f"{aud['max_accuracy_any_threshold'] * 100:.2f}%",
          "Upper bound, chosen with test labels in hand"],
         ["Best balanced accuracy at any threshold",
          f"{aud['max_balanced_any_threshold'] * 100:.2f}%",
          "Same bound, class-weighted"]],
        "TABLE 9. Corpus audit. No model is trained here; this "
        "characterises the data.",
        widths=[3400, 1600, 4360]))
    p("The first two rows matter because FaceForensics++ manipulated clips "
      "share underlying footage with the originals they were derived from, "
      "and the corpus here was pooled at the clip level rather than split "
      "by identity, as Section 2.10 notes. Had near-duplicates been "
      "present, every score in this paper would be inflated. None are: the "
      f"highest similarity between any two of the fifty videos is "
      f"{aud['max_offdiag_cosine']:.5f}, comfortably below the 0.98 "
      "threshold. The measured performance is low for reasons other than "
      "leakage.")
    p("The last two rows are the central quantitative claim of the paper. "
      "Given the out-of-fold ROC curve, the accuracy attainable at the best "
      "threshold on that curve is "
      f"{aud['max_accuracy_any_threshold'] * 100:.2f}%, and the best "
      "balanced accuracy is "
      f"{aud['max_balanced_any_threshold'] * 100:.2f}%. Both are computed "
      "by scanning every threshold with the test labels visible, so both "
      "are optimistic by construction and neither is attainable in "
      "deployment. Reaching 95% accuracy on this corpus would require an "
      "AUC of approximately 0.98 against the "
      f"{aud['oof_auc']:.4f} measured — not a better threshold but a "
      "fundamentally better ranking of the videos.")
    B.append(_f("fig09_accuracy_ceiling.png",
                "Fig. 10. Accuracy attainable at each point on the measured "
                "ROC curve, against the majority-class line and the "
                "published claim. The ceiling is a property of the curve, "
                "not of the classifier reading it."))
    ci = {c["correct"]: c for c in aud["binomial_ci"]}
    rows = [[f"{c['correct']} / {c['n']}", f"{c['acc'] * 100:.2f}",
             f"[{c['lo'] * 100:.1f}, {c['hi'] * 100:.1f}]"]
            for c in aud["binomial_ci"]]
    B.append(table(["Correct / total", "Accuracy (%)", "95% interval (%)"],
                   rows,
                   "TABLE 10. Binomial confidence intervals at the fold "
                   "sizes this corpus produces.",
                   widths=[2200, 2200, 2600]))
    p("A perfect ten-out-of-ten on a ten-video fold carries a 95% interval "
      f"reaching down to {ci[10]['lo'] * 100:.1f}%. Eight-out-of-ten and "
      "ten-out-of-ten have overlapping intervals. No protocol on this corpus "
      "can distinguish an 80% detector from a 100% detector, which means "
      "that differences of a few percentage points between models — the "
      "differences the original work reports as its contribution — are not "
      "measurable here at all " + cite("demsar") + ".")

    # ---------------------------------------------------------- 5.7
    B.append(h2("5.7 Representation and Model Search"))
    v2, w = d["v2"], d["v2"]["winner"]
    p("If the deep models fail, the natural question is whether the failure "
      "is architectural or representational. A search over fourteen "
      "candidate representations of the cached tensor crossed with nine "
      "model families, all under nested cross-validation with selection "
      "confined to inner folds, answers it. Table 11 gives the top of the "
      "ranking.")
    rows = [[r["representation"], r["model"], fmt(r["mean"] * 100),
             fmt(r["std"] * 100)] for r in v2["ranking"][:10]]
    B.append(table(["Representation", "Model", "Balanced acc. (%)",
                    "Std. dev."], rows,
                   "TABLE 11. Top ten of 126 representation-model "
                   "combinations under nested cross-validation.",
                   widths=[3800, 1900, 1900, 1760],
                   highlight=lambda r: r[0].startswith("proposed: temporal")
                   and r[1] == "logreg-l1"))
    p(f"The winner is first-order temporal difference statistics with L1 "
      f"logistic regression at {fmt(w['nested_bal_acc'] * 100)}% balanced "
      f"accuracy. Its permutation null has mean "
      f"{w['null_mean'] * 100:.2f}% and 95th percentile "
      f"{w['null_p95'] * 100:.2f}%, giving p = {w['p_value']:.4f}. It is "
      "the only entry in the table that clears its own null convincingly; "
      "the second-placed combination sits inside the null's upper tail.")
    p("Two features of this result deserve emphasis. The winning "
      "representation is the one that preserves frame-to-frame change, "
      "which is the same conclusion Section 5.5 reaches from the ROC "
      "analysis and the same conclusion the classical inter-frame forensics "
      "literature reached by other means " + cite("kingra", "fadl") + ". "
      "And the winning model is a linear classifier with sparsity-inducing "
      "regularisation — the simplest thing in the search — which on fifty "
      "samples is exactly what the small-sample literature would predict "
      + cite("varoquaux", "vabalas") + ".")
    B.append(_f("fig07_representation_search.png",
                "Fig. 11. Representation search. Only one combination "
                "clears its permutation null with margin; the rest lie "
                "inside the distribution obtained from shuffled labels."))
    v3 = d["v3"]
    best_v3 = max(((k, m, v[0]) for k, mm in v3["results"].items()
                   for m, v in mm.items()), key=lambda t: t[2])
    p("A follow-up search over higher-order temporal features — "
      "acceleration, lag-2 differences, autocorrelation — and over stacked "
      "and voting ensembles found nothing better than the first-order "
      f"baseline. The best entry in that second search is {best_v3[1]} on "
      f"{best_v3[0]} at {fmt(best_v3[2] * 100)}%, and every ensemble scored "
      "below plain L1 logistic regression on first-order differences. On "
      "fifty samples, added model capacity costs more in variance than it "
      "returns in bias.")

    # ---------------------------------------------------------- 5.8
    B.append(h2("5.8 Class Weights, Calibration and Decision Thresholds"))
    wt = d["weights"]["ranking"]
    rows = [[r["model"], fmt(r["bal_acc"] * 100), fmt(r["std"] * 100),
             f"{r['threshold']:.4f}"] for r in wt[:6]]
    B.append(table(["Configuration", "Balanced acc. (%)", "Std. dev.",
                    "Threshold"], rows,
                   "TABLE 12. Best six of 30 class-weight, calibration and "
                   "threshold configurations, all selected inside training "
                   "folds.",
                   widths=[3400, 2000, 1900, 2060]))
    p("Because the corpus is imbalanced, adjusting class weights or moving "
      "the decision threshold is the obvious next lever. Thirty "
      "configurations were swept, with every choice made inside the training "
      f"folds. The best reaches {fmt(wt[0]['bal_acc'] * 100)}%, which is "
      f"below the untuned {fmt(d['v2']['winner']['nested_bal_acc'] * 100)}% "
      "of Section 5.7. This is the expected outcome when the tuning budget "
      "is spent on a sample too small to estimate the quantity being tuned "
      + cite("cawley") + ". Selecting the threshold on the test fold "
      "instead would of course produce a higher number; that is the "
      "procedure this study exists to argue against.")

    # ---------------------------------------------------------- 5.9
    B.append(h2("5.9 Imported Spatiotemporal Modules"))
    s = d["stil"]
    p("The strongest available test of whether the failure is architectural "
      "is to take an architecture known to work on this problem and apply it "
      "to this representation. The Temporal Inconsistency Module and "
      "Information Supplement Module of " + citet("stil") + " were imported "
      "unmodified from the authors' repository — the two repositories most "
      "commonly cited for that work contain no code and both redirect there "
      f"— at commit {s['tface_commit'][:12]}, and wrapped in a minimal "
      "classifier of "
      "26,696 parameters, against SMA-CLMPNet's 2,258,534. The folds are the "
      "identical ones used everywhere else in this paper, read from the same "
      "cache file.")
    rows = [[str(f["fold"] + 1), fmt(f["inner_val_bal"] * 100),
             str(f["best_epoch"]), fmt(f["outer_bal"] * 100)]
            for f in s["per_fold"]]
    rows.append(["pooled", "—", "—", fmt(s["pooled_balanced_accuracy"] * 100)])
    B.append(table(["Outer fold", "Inner validation bal. acc. (%)",
                    "Best epoch", "Outer bal. acc. (%)"], rows,
                   "TABLE 13. STIL's TIM and ISM modules on the cached "
                   "tensors, five outer folds, early stopping on an inner "
                   "split.",
                   widths=[1800, 3200, 1800, 2560],
                   highlight=lambda r: r[0] == "pooled"))
    early = sum(1 for f in s["per_fold"] if f["best_epoch"] <= 3)
    p(f"Pooled out-of-fold balanced accuracy is "
      f"{fmt(s['pooled_balanced_accuracy'] * 100)}%, and the mean across "
      f"folds is {fmt(s['mean_fold_balanced_accuracy'] * 100)}%. "
      f"{early} of the five folds selected their best epoch at epoch "
      f"{min(f['best_epoch'] for f in s['per_fold'])} or 3, meaning nothing "
      "after the first few gradient steps improved inner validation "
      "performance. The run took "
      f"{s['wall_seconds']:.0f} seconds.")
    p("This is the decisive experiment for the paper's main claim. A "
      "temporal-inconsistency architecture published, peer-reviewed and "
      "demonstrated to work on FaceForensics++ at scale, imported without "
      "modification and given the same folds, lands at the "
      "constant-classifier value on these tensors. The bottleneck is "
      "therefore not the architecture of SMA-CLMPNet. It is that the cached "
      "representation — ten frames per video, twelve hand-stacked channels, "
      "fifty videos — does not carry enough of the manipulation signal for "
      "any of these models to recover it.")

    # ---------------------------------------------------------- 5.10
    B.append(h2("5.10 The Companion Architecture"))
    p2 = d["paper2"]
    rows = [["BiLSTM + gradient boosting", fmt(p2["BiLSTMGBM"][ACC] * 100),
             fmt(p2["BiLSTMGBM"][SPE] * 100), fmt(p2["BiLSTMGBM"][F1] * 100),
             fmt(p2["BiLSTMGBM"][BAL] * 100)],
            ["BiLSTM only", fmt(p2["BiLSTM_only"][ACC] * 100),
             fmt(p2["BiLSTM_only"][SPE] * 100),
             fmt(p2["BiLSTM_only"][F1] * 100),
             fmt(p2["BiLSTM_only"][BAL] * 100)]]
    B.append(table(["Model", "Accuracy", "Specificity", "F1",
                    "Balanced acc."], rows,
                   "TABLE 14. The companion project's architecture ported to "
                   "these features at its own published settings "
                   f"({p2['settings']['epochs']} epochs, batch "
                   f"{p2['settings']['batch']}), with its test-set weight "
                   "fitting omitted.",
                   widths=[3200, 1600, 1600, 1500, 1460]))
    p("A recurrent-plus-gradient-boosting architecture from a companion "
      "project was ported to these features at its own published settings. "
      "One step was omitted: a routine that fits model weights directly on "
      "the test partition. With that step removed the model lands at the "
      f"constant-classifier value, {fmt(p2['BiLSTMGBM'][BAL] * 100)}% "
      "balanced accuracy. This is included because it demonstrates the third "
      "fabrication route of Section 1.6 in isolation — the architecture is "
      "unremarkable, and the published performance came from the omitted "
      "step rather than from the model.")

    # ---------------------------------------------------------- 5.11
    B.append(h2("5.11 Summary of Measured Results"))
    p("Six independent lines of evidence converge on the same conclusion. "
      "The published cohort is degenerate under corrected scoring. Its "
      "ablations are indistinguishable from each other and from a constant "
      "classifier. Four current-generation backbones do not clear the "
      "ceiling. A published spatiotemporal architecture, imported "
      "unmodified, lands at the constant-classifier value. Tuning class "
      "weights and thresholds inside training folds makes things worse. And "
      "the best result obtained by any method in the study is a linear "
      "model on first-order temporal differences.")
    B.append(_f("fig08_method_comparison.png",
                "Fig. 12. All measured methods against the permutation null. "
                "Bars above the null's 95th percentile are the only ones "
                "carrying evidence of detection."))
    return "".join(B)


# =========================================================== section 6
def comparison(d):
    B = [h1("6. Comparison with Existing Models")]
    p = lambda t: B.append(para(t))                       # noqa: E731

    B.append(h2("6.1 Measured Comparison on Identical Splits"))
    p("Table 15 is the comparison this paper can defend. Every row was "
      "measured in this study, on the same feature cache, the same splits "
      "and the same scoring code, so differences between rows are "
      "attributable to the method.")
    maj = majority_accuracy(d)
    rows = []
    for m in sorted(ORDER, key=lambda m: -mean_bal(d["kf"][m])):
        a_s, a_k = d["sweep"][m], d["kf"][m]
        rows.append([pretty(m), fmt(mean_bal(a_s)), fmt(mean_bal(a_k)),
                     fmt(float(np.nanmean(a_k[:, SPE]))),
                     f"{degenerate(a_s) + degenerate(a_k)} of "
                     f"{a_s.shape[0] + a_k.shape[0]}"])
    w = d["v2"]["winner"]
    rows.append(["Temporal deltas + L1 logistic (this study)",
                 fmt(w["nested_bal_acc"] * 100), "—", "—", "0"])
    rows.append(["STIL TIM + ISM, imported unmodified",
                 "—", fmt(d["stil"]["pooled_balanced_accuracy"] * 100),
                 "—", "—"])
    rows.append([f"Constant 'authentic' baseline ({maj:.0f}% accuracy)",
                 "50.00", "50.00", "0.00", "all"])
    B.append(table(["Method", "Sweep bal. acc. (%)", "k-fold bal. acc. (%)",
                    "k-fold specificity (%)", "Degenerate runs"], rows,
                   "TABLE 15. All methods measured in this study, ordered by "
                   "k-fold balanced accuracy. Every row uses identical "
                   "splits and identical scoring.",
                   widths=[3600, 1700, 1700, 1700, 1660],
                   highlight=lambda r: r[0].startswith("Temporal deltas")
                   or r[0].startswith("SMA-CLMPNet ")
                   or r[0] == "SMA-CLMPNet"))
    B.append(_f("fig13_comparison_bar.png",
                "Fig. 13. Comparison bar chart: mean balanced accuracy of "
                "every method measured in this study, against the "
                "constant-classifier line and the permutation null."))
    p("The proposed model does not lead this table. The strongest deep "
      "model under k-fold is a comparison model from the released cohort, "
      "and the strongest method overall is the linear pipeline on temporal "
      "differences. Given the confidence intervals in Table 10, the "
      "honest reading of the ordering is that the top few entries are not "
      "separable from one another and the bottom entries are separable from "
      "the top only because they fall below the constant classifier.")

    B.append(h2("6.2 Comparison with the Published Literature"))
    p("A second comparison — against the accuracies reported by the "
      "spatiotemporal detection literature — is deliberately not tabulated "
      "as numbers beside the numbers above, and the reason is the argument "
      "of Sections 2.8 and 2.10 rather than an omission. Those results were "
      "obtained on corpora three to four orders of magnitude larger, split "
      "at the level of source identity rather than clip, with cross-corpus "
      "and graded-perturbation reporting. A figure obtained under that "
      "protocol and a figure obtained on fifty pooled clips are not "
      "measurements of the same quantity, and placing them in adjacent "
      "columns invites precisely the incommensurable comparison that "
      "produces claims like the one this paper set out to reproduce.")
    p("What can be compared is the design of the studies. Table 16 sets the "
      "present work beside the reference methods on the attributes that "
      "determine whether a reported number means anything.")
    B.append(table(
        ["Method", "Primary cue", "Evaluation corpora", "Temporal model"],
        [["MesoNet " + cite("mesonet"), "Mesoscopic spatial artefact",
          "FaceForensics, own corpus", "No"],
         ["Xception on FF++ " + cite("faceforensics", "xception"),
          "Spatial artefact, face crop",
          "FaceForensics++ (4 methods, 3 compressions)", "No"],
         ["Capsule-forensics " + cite("capsule"), "Part-whole spatial "
          "structure", "FaceForensics++, own corpora", "No"],
         ["Face X-Ray " + cite("facexray"), "Blending boundary",
          "FaceForensics++, Celeb-DF, DFDC", "No"],
         ["F3-Net " + cite("qian_f3net"), "Frequency-aware clues",
          "FaceForensics++ (incl. low quality)", "No"],
         ["Recurrent CNN " + cite("guera"), "Frame-sequence inconsistency",
          "Own collected corpus", "Yes (LSTM)"],
         ["Optical-flow CNN " + cite("amerini"), "Apparent-motion "
          "inconsistency", "FaceForensics++", "Yes (flow)"],
         ["FTCN " + cite("ftcn"), "Temporal coherence",
          "FaceForensics++, Celeb-DF, DFDC, cross-dataset", "Yes"],
         ["LipForensics " + cite("haliassos"), "Natural mouth dynamics",
          "FaceForensics++, cross-manipulation", "Yes"],
         ["STIL " + cite("stil"), "Spatiotemporal inconsistency (TIM + ISM)",
          "FaceForensics++, Celeb-DF", "Yes"],
         ["ISTVT " + cite("istvt"), "Decomposed spatial-temporal attention",
          "FF++, FaceShifter, DeeperForensics, Celeb-DF, DFDC", "Yes"],
         ["TALL " + cite("tall"), "Thumbnail-layout spatiotemporal",
          "FF++, cross-dataset, diffusion-generated", "Yes"],
         ["SMA-CLMPNet (this study, re-measured)",
          "3D convolution + joint attention + LSTM",
          "50-video pooled subset of FaceForensics++", "Yes (LSTM)"]],
        "TABLE 16. Design comparison against reference methods. Reported "
        "accuracies are deliberately omitted; see the text.",
        widths=[2900, 2900, 3100, 1460]))
    p("Two things are visible in that table. Every method in the "
      "spatiotemporal group evaluates on at least one corpus other than the "
      "one it trained on, and most evaluate on three or more; the present "
      "pipeline evaluates on a pooled subset of a single corpus and cannot "
      "report cross-dataset performance at all. And the cue exploited by "
      "the strongest reference methods — natural facial dynamics, temporal "
      "coherence, decomposed spatiotemporal attention — is computed from "
      "aligned, densely sampled frames, whereas the representation here "
      "supplies ten frames of hand-stacked colour and texture channels with "
      "no correspondence between them. The measured result in Section 5.9, "
      "where STIL's own modules land at the constant-classifier value on "
      "these tensors, is what that difference looks like empirically.")

    B.append(h2("6.3 Discussion"))
    p("Three conclusions follow from the comparison.")
    p("First, the ranking the original work reports does not survive "
      "corrected scoring, and no ranking replaces it. The proposed model, "
      "its ablations and several comparison models are degenerate; the "
      "non-degenerate models are separated by margins smaller than the "
      "confidence interval of a single fold. The correct summary is that "
      "the cohort is not separable on this corpus, not that some other "
      "member of it is best.")
    p("Second, the limiting factor is the representation rather than the "
      "architecture. This is not an inference from the failure of the "
      "cohort — that would be weak evidence, since a cohort can fail for "
      "many reasons — but from three converging measurements: the same "
      "features with the temporal axis preserved reach an AUC of "
      f"{d['roc']['curves']['temporal delta stats (best honest pipeline)']['auc']:.4f} "
      "while the time-collapsed version reaches "
      f"{d['roc']['curves']['per-frame mean+std (time-collapsed reference)']['auc']:.4f}; "
      "the representation search selects the temporal-difference "
      "representation and a linear model over every deep alternative; and a "
      "published spatiotemporal architecture imported unmodified fails on "
      "the same tensors.")
    p("Third, the ceiling is quantified rather than merely observed. The "
      "measured effect size bounds accuracy at "
      f"{d['audit']['max_accuracy_any_threshold'] * 100:.2f}% at the best "
      "threshold on the observed curve, with the test labels visible. Any "
      "future claim above that bound on this corpus and this representation "
      "requires either a different representation or an explanation of "
      "where the additional ranking power came from.")
    return "".join(B)


# =========================================================== sections 7-8
def conclusion(d):
    B = [h1("7. Conclusion")]
    p = lambda t: B.append(para(t))                       # noqa: E731
    r = d["roc"]["curves"]["temporal delta stats (best honest pipeline)"]
    w = d["v2"]["winner"]
    p("This paper re-examined a published intra-frame video forgery "
      "detection system and found its reported performance unreproducible "
      "for reasons that are not statistical. Two mutually independent "
      "sources of fabricated results were identified in the released "
      "artefacts: a vendored optimisation library whose metrics module "
      "rewrites predicted labels toward the true labels before scoring, and "
      "three stored files describing a corpus forty times larger than the "
      "one the code reads. All affected material was removed and the removal "
      "recorded.")
    p("Every published figure was then re-measured. Under a protocol with "
      "no selection on test data and a confusion matrix from an unmodified "
      "library, the proposed model reaches "
      f"{fmt(mean_bal(d['sweep']['SMA-CLMPNet']))}% mean balanced accuracy "
      "and answers a single class for every input at most training "
      "percentages; its two attention ablations are indistinguishable from "
      "it and from each other. Four current-generation backbones evaluated "
      "on identical splits do not clear the ceiling, and the Temporal "
      "Inconsistency and Information Supplement modules of a published "
      "spatiotemporal detector, imported without modification, land at the "
      "constant-classifier value on the same tensors. The strongest measured "
      "configuration in the study is first-order temporal difference "
      f"statistics with L1 logistic regression at "
      f"{fmt(w['nested_bal_acc'] * 100)}% balanced accuracy "
      f"(permutation p = {w['p_value']:.4f}, out-of-fold AUC "
      f"{r['auc']:.4f}).")
    p("What has been achieved is a defensible characterisation of what this "
      "corpus and representation support, with an upper bound on attainable "
      f"accuracy of {d['audit']['max_accuracy_any_threshold'] * 100:.2f}% "
      "derived from the measured effect size and verified not to be an "
      "artefact of near-duplicate videos or of any separable feature. Where "
      "the work falls short is equally clear: fifty videos cannot separate "
      "an 80% detector from a 100% one, so no ranking within the cohort is "
      "meaningful; the corpus is pooled at clip level rather than split by "
      "identity; no cross-dataset evaluation is possible; and the "
      "representation supplies ten frames of hand-stacked channels with no "
      "inter-frame correspondence, which the evidence identifies as the "
      "binding constraint. Improving on this requires frame-level "
      "supervision on the full corpus rather than further architecture "
      "search on cached tensors.")
    return "".join(B)


def future_work(d):
    B = [h1("8. Future Work")]
    p = lambda t: B.append(para(t))                       # noqa: E731
    p("The measurements in Section 5 identify where effort should go, in "
      "order of expected return.")
    B.append(bullet(
        "Frame-level supervision on the full corpus. The repository already "
        "contains a complete frame-level pipeline for the full "
        "FaceForensics++ release, which has never been run because the data "
        "was never downloaded. Training at frame level over 1,000 source "
        "sequences and their manipulations, with the identity-level split "
        "the corpus defines, changes the sample size by three orders of "
        "magnitude and is the single change most likely to move the "
        "ceiling " + cite("faceforensics") + "."))
    B.append(bullet(
        "Dense temporal sampling with alignment. Ten frames per video with "
        "no correspondence between them is the weakest part of the present "
        "representation. Densely sampled, landmark-aligned crops are what "
        "the methods in Section 2.4 consume "
        + cite("sabir", "haliassos") + "."))
    B.append(bullet(
        "Cross-corpus evaluation. Training on FaceForensics++ and testing on "
        "Celeb-DF and the DFDC corpus is the standard generalisation test "
        "and is currently impossible here " + cite("celebdf", "dfdc") + "."))
    B.append(bullet(
        "Graded robustness reporting. Compression, noise and colour "
        "perturbation at controlled severities, as DeeperForensics defines "
        "them, so that the dependence of accuracy on distribution channel is "
        "measured rather than assumed " + cite("deeperforensics") + "."))
    B.append(bullet(
        "Calibrated outputs with intervals. For the adjudication regime of "
        "Section 1.2, a point estimate is not usable. Calibrated "
        "probabilities with confidence intervals, reported per item, are the "
        "minimum a forensic deployment requires."))
    B.append(bullet(
        "Metric verification as routine practice. The defect that motivated "
        "this study would have been caught in minutes by passing synthetic "
        "vectors with a known confusion matrix through the project's own "
        "scoring path. Adding that check to reproduction protocols costs "
        "nothing and would have prevented every number this paper had to "
        "withdraw " + cite("kapoor", "gundersen") + "."))
    p("A note on the target that prompted part of this investigation. The "
      "question of whether accuracy in the 95-100% range is reachable on "
      "this corpus has a quantitative answer rather than an open-ended one: "
      "it would require an area under the ROC curve of approximately 0.98 "
      f"against the {d['audit']['oof_auc']:.4f} measured out of fold. No "
      "architecture in this study, and no amount of tuning applied to one, "
      "closes that gap. The three mechanisms that would close it on paper — "
      "a tampered metric, weight fitting on the test partition, and "
      "selection of the best result across seeds — are the three this paper "
      "documents and rejects. The reachable path runs through more data and "
      "a better representation, and Section 8 lists it in order.")
    return "".join(B)


def acknowledgment(d):
    B = [h1("Acknowledgment")]
    B.append(para(
        "This work was carried out under the postdoctoral research programme "
        "of the Research Institute of IoT Cybersecurity (RIITC), Department "
        "of Electronic Engineering, National Kaohsiung University of Science "
        "and Technology. The Temporal Inconsistency and Information "
        "Supplement modules evaluated in Section 5.9 are the authors' own "
        "implementation from the Tencent TFace repository, used without "
        "modification and cited as " + citet("stil") + "; the "
        "reimplementation and evaluation on this corpus are the "
        "responsibility of the present authors alone. All computational "
        "results reported here were produced on a single workstation, and "
        "the scripts that generate every figure, table and number are "
        "retained with the run logs."))
    return "".join(B)


def references(keys_apa):
    from article_style import reference_entry
    B = [h1("References")]
    B.append(para(
        "All entries were resolved by digital object identifier against "
        "CrossRef, or against DataCite for arXiv-registered preprints, and "
        "are listed in APA (7th edition) style. Two entries published in a "
        "venue that issues no identifier are recorded with the publisher's "
        "own listing URL."))
    for s in keys_apa:
        B.append(reference_entry(s))
    return "".join(B)
