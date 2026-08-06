# Run logs

Every log behind the results. Repetitive noise (sklearn warnings, Keras progress bars, TensorFlow device probes) is collapsed into counted placeholders; every informative line is kept.

Every log here records a run scored by `../Optimized/metrics_fixed.py`, or a diagnostic. The three console records of runs scored by the tampered `mealpy/metrics.py` were removed on 2026-08-06 along with the rest of the fabricated material — see [`../Optimized/PROVENANCE.md`](../Optimized/PROVENANCE.md).

| Log | Lines | Collapsed | What it records |
|---|---|---|---|
| [`conda_pytorch_ssl_failure.log`](conda_pytorch_ssl_failure.log) | 114 | — | conda install pytorch FAILED with CERTIFICATE_VERIFY_FAILED through the network's TLS interception; why torch stayed unavailable and the SubFunctions __init__ bypass was needed. |
| [`final_tables.log`](final_tables.log) | 13 | — | Full metric tables by training percentage for the best honest pipeline. |
| [`frame_embeddings.log`](frame_embeddings.log) | 31 | — | Per-frame and frame-difference backbone embeddings of the 'proposed' tensor (MobileNetV3Large 15,360-dim, EfficientNetV2S 20,480-dim). |
| [`git_push_initial.log`](git_push_initial.log) | 26 | — | Initial push of 1,672 files to GitHub. |
| [`keras_weight_download.log`](keras_weight_download.log) | 21 | — | First import of SubFunctions: ResNet101 (180 MB) and VGG16 (553 MB) download at module scope - the 733 MB cost of a bare import. |
| [`kfold_true.log`](kfold_true.log) | 4841 | — | K-fold comparison, k = 6..10, stratified folds, correct scoring. Feeds section 5.6.2. The published KFAnalysis could not be used: Analysis.py:355 indexes data['image'], a key ReadDataset never stores.  **(still being written)** |
| [`kfold_true_interrupted.log`](kfold_true_interrupted.log) | 775 | — | First k-fold attempt. Died when the session that launched it exited, 4 of 12 models into k=6 and before the first checkpoint. |
| [`kfold_true_interrupted2.log`](kfold_true_interrupted2.log) | 1245 | — | Second k-fold attempt. Died 6 models into k=6 when the agent task that had called Start-Process was reaped - Start-Process does not escape the harness job object. Fixed by launching through WMI (Optimized/run_kfold.bat) and adding --resume to kfold mode. |
| [`optimize_v2.log`](optimize_v2.log) | 72 | — | Representation and model search: 14 representations x 9 model families under nested cross-validation, with a 100-shuffle permutation test on the winner. Established the temporal-delta signal at p = 0.0099. |
| [`optimize_v2_frames.log`](optimize_v2_frames.log) | 5108 | — | Second search pass over the per-frame backbone embeddings, which the first pass missed because they were written after it began. |
| [`optimize_v3.log`](optimize_v3.log) | 47 | — | Higher-order temporal features (acceleration, lag-2, autocorrelation) and stacked/voting ensembles. Every addition scored BELOW plain L1 logistic regression on first-order deltas. |
| [`optimize_weights.log`](optimize_weights.log) | 47 | — | Class-weight, probability-calibration and decision-threshold sweep, 30 configurations, all selected inside training folds. Best 69.67%, below the untuned 77.17%. |
| [`paper2_model_500.log`](paper2_model_500.log) | 16 | — | Paper 2's BiLSTMGBM ported to Paper 1's features at its own settings (500 epochs, batch 32, incremental learning), omitting the test-set weight fitting. Lands at chance. |
| [`stil_tim.log`](stil_tim.log) | 49 | — | STIL's Temporal Inconsistency Module (Gu et al., ACM MM 2021) on this feature tensor. TIM_Module and ISM_Module imported unmodified from Tencent/TFace 171ec143 - the two repos everyone cites for STIL contain no code and both redirect there. 26,696 parameters against SMA-CLMPNet's 2,258,534. Pooled out-of-fold balanced accuracy 50.49%, catching 6 of 21 forgeries; three of five folds checkpointed at epoch 1 or 3, so nothing after initialisation improved validation. |
| [`sweep_true.log`](sweep_true.log) | 6550 | — | The corrected re-evaluation. All 7 of the paper's models plus the 4 current-generation backbones and SMA-CLMPNet-Opt, across training percentages 40-90%, scored with a real confusion matrix. 10,109 s. Source of Analysis1/TRUE and of the tables in sections 5.6.1 / 5.8. |
| [`webapp_server.log`](webapp_server.log) | 59 | — | Flask dev server serving the analyze API during verification. |
| [`webapp_start_failure.log`](webapp_start_failure.log) | 22 | — | First webapp launch: AttributeError, module 'cv2' has no attribute 'data'. Fixed by bundling the Haar cascade. |
| [`session_notes.md`](session_notes.md) | 88 | — | Running notes: every error hit and the fix. |

Result arrays are in `../Analysis1/TRUE` (training-percentage sweep) and `../Analysis1/TRUE_KF` (k-fold); search results are the JSON files in `../Optimized/`.

Regenerate this index with `python Optimized/save_logs.py`.
