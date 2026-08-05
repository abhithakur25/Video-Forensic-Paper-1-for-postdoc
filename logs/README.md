# Run logs

Every log behind the results. Repetitive noise (sklearn warnings, Keras progress bars, TensorFlow device probes) is collapsed into counted placeholders; every informative line is kept.

Logs marked SUPERSEDED were produced by the tampered metric in `mealpy/metrics.py` and are retained as the record of what the pipeline emits as delivered — see [`../Optimized/INTEGRITY_FINDING.md`](../Optimized/INTEGRITY_FINDING.md).

| Log | Lines | Collapsed | What it records |
|---|---|---|---|
| [`conda_pytorch_ssl_failure.log`](conda_pytorch_ssl_failure.log) | 110 | — | conda install pytorch FAILED with CERTIFICATE_VERIFY_FAILED through the network's TLS interception; why torch stayed unavailable and the SubFunctions __init__ bypass was needed. |
| [`evaluation_kfold.log`](evaluation_kfold.log) | 211 | — | SUPERSEDED. Partial k-fold under the tampered metric; stopped once the fabrication was found. |
| [`evaluation_kfold_aborted.log`](evaluation_kfold_aborted.log) | 192 | — | First k-fold attempt at 2 folds per k, aborted at 12/70 fits on a projected ~8.75 h runtime. |
| [`evaluation_tp_sweep.log`](evaluation_tp_sweep.log) | 747 | — | SUPERSEDED. The original training-percentage sweep, scored by the tampered metric. Kept as the record of what the pipeline produces as delivered. |
| [`final_tables.log`](final_tables.log) | 9 | — | Full metric tables by training percentage for the best honest pipeline. |
| [`frame_embeddings.log`](frame_embeddings.log) | 27 | 5 | Per-frame and frame-difference backbone embeddings of the 'proposed' tensor (MobileNetV3Large 15,360-dim, EfficientNetV2S 20,480-dim). |
| [`git_push_initial.log`](git_push_initial.log) | 22 | — | Initial push of 1,672 files to GitHub. |
| [`keras_weight_download.log`](keras_weight_download.log) | 17 | — | First import of SubFunctions: ResNet101 (180 MB) and VGG16 (553 MB) download at module scope - the 733 MB cost of a bare import. |
| [`kfold_true.log`](kfold_true.log) | 439 | — | K-fold comparison, k = 6..10, stratified folds, correct scoring. Feeds section 5.6.2. The published KFAnalysis could not be used: Analysis.py:355 indexes data['image'], a key ReadDataset never stores.  **(still being written)** |
| [`optimize_v2.log`](optimize_v2.log) | 68 | 87928 | Representation and model search: 14 representations x 9 model families under nested cross-validation, with a 100-shuffle permutation test on the winner. Established the temporal-delta signal at p = 0.0099. |
| [`optimize_v2_frames.log`](optimize_v2_frames.log) | 5104 | 6560 | Second search pass over the per-frame backbone embeddings, which the first pass missed because they were written after it began. |
| [`optimize_v3.log`](optimize_v3.log) | 43 | — | Higher-order temporal features (acceleration, lag-2, autocorrelation) and stacked/voting ensembles. Every addition scored BELOW plain L1 logistic regression on first-order deltas. |
| [`optimize_weights.log`](optimize_weights.log) | 43 | — | Class-weight, probability-calibration and decision-threshold sweep, 30 configurations, all selected inside training folds. Best 69.67%, below the untuned 77.17%. |
| [`paper2_model_500.log`](paper2_model_500.log) | 12 | 2 | Paper 2's BiLSTMGBM ported to Paper 1's features at its own settings (500 epochs, batch 32, incremental learning), omitting the test-set weight fitting. Lands at chance. |
| [`sweep_true.log`](sweep_true.log) | 6546 | 262 | The corrected re-evaluation. All 7 of the paper's models plus the 4 current-generation backbones and SMA-CLMPNet-Opt, across training percentages 40-90%, scored with a real confusion matrix. 10,109 s. Source of Analysis1/TRUE and of the tables in sections 5.6.1 / 5.8. |
| [`webapp_server.log`](webapp_server.log) | 55 | — | Flask dev server serving the analyze API during verification. |
| [`webapp_start_failure.log`](webapp_start_failure.log) | 18 | — | First webapp launch: AttributeError, module 'cv2' has no attribute 'data'. Fixed by bundling the Haar cascade. |
| [`session_notes.md`](session_notes.md) | 88 | — | Running notes: every error hit and the fix. |

Result arrays are in `../Analysis1/TRUE` (training-percentage sweep) and `../Analysis1/TRUE_KF` (k-fold); search results are the JSON files in `../Optimized/`.

Regenerate this index with `python Optimized/save_logs.py`.
