# Run logs

Captured from the session that built the harness, ran the evaluation, and deployed the web app.

| Log | Lines | What it records |
|---|---|---|
| [`evaluation_tp_sweep.log`](evaluation_tp_sweep.log) | 743 | Training-percentage comparative sweep, 40-90%, 10 epochs, 7 models. COMPLETED in 5639 s -> Analysis1/TP/COM_A..H.npy |
| [`evaluation_kfold.log`](evaluation_kfold.log) | 204 | K-fold comparative analysis, k=6..10, 1 fold per k, 10 epochs. Feeds section 5.6.2 of the paper. |
| [`evaluation_kfold_aborted.log`](evaluation_kfold_aborted.log) | 188 | First k-fold attempt at 2 folds per k. ABORTED at 12/70 fits: projected ~8.75 h because k-fold always trains on 41-45 of the 50 samples, ~3x slower per fit than the TP sweep. |
| [`keras_weight_download.log`](keras_weight_download.log) | 12 | First import of SubFunctions: ResNet101 (180 MB) and VGG16 (553 MB) download at module scope. Shows the 733 MB cost of a bare import. |
| [`conda_pytorch_ssl_failure.log`](conda_pytorch_ssl_failure.log) | 106 | conda install pytorch - FAILED. CERTIFICATE_VERIFY_FAILED via the network's TLS interception; why torch stayed unavailable. |
| [`git_push_initial.log`](git_push_initial.log) | 18 | Initial push of 1672 files to GitHub. |
| [`webapp_start_failure.log`](webapp_start_failure.log) | 14 | First webapp launch - AttributeError: module 'cv2' has no attribute 'data'. Fixed by bundling the Haar cascade. |
| [`webapp_server.log`](webapp_server.log) | 51 | Flask dev server serving the analyze API during verification. |
| [`session_notes.md`](session_notes.md) | - | Running notes: every error hit and the fix that resolved it. |

Keras progress-bar lines (`[====] - ETA:`) are collapsed; the summary lines that report totals and timings are kept.

Evaluation result tables live in `../driver_out/`; the numeric arrays they came from are in `../Analysis1/`.
