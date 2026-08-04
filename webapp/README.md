# VForensiQ — web front end for Paper 1

Browser interface for the SMA-CLMPNet video forgery pipeline
(*Design and Development of a Video Forgery Model Using Deep Learning with
Attention Mechanisms*). Flask + OpenCV, designed to deploy on **Render**.

Styled to match the sibling [ForensiQ](https://forensiq-detector.onrender.com)
image-forensics app: dark forensic theme, single-purpose upload flow, per-stage
result cards.

---

## What it actually runs

| Stage | Source in the research code | Runtime |
|---|---|---|
| Gradient key-frame selection | `GetPreprocessing.py` | OpenCV Sobel |
| Viola-Jones face ROI | `GUI.roi()` | Haar cascade (bundled in `assets/`) |
| Lucas-Kanade optical flow | `GUI.object_flow_features()` | OpenCV |
| LDZP texture | `SubFunctions/LDZP.py` | 8 directional kernels |
| 3×3 neighbourhood statistics | `GetFeatures.statistical_features()` | vectorised NumPy/SciPy |
| GLCM descriptors | `GetFeatures.py` | NumPy |
| High-pass residual | — | OpenCV |

`statistical_features()` in the research code loops over every pixel in Python;
that is far too slow for a request, so the app computes the identical
8-neighbourhood statistics vectorised.

### What it deliberately does not do

**No forged/authentic verdict from the published network.** There is no trained
SMA-CLMPNet checkpoint anywhere in the repository — `Analysis.py` trains every
model on the fly and persists only metrics. Rather than invent a verdict, the app
reports a **tamper-evidence index**: three measured terms, each normalised to
0–1 and combined with fixed weights that are returned in every API response so
the number can be audited.

| Term | Weight | Measured from |
|---|---|---|
| Key-frame discontinuity | 0.40 | peak z-score of inter-frame gradient change |
| Optical-flow irregularity | 0.35 | peak z-score of mean flow magnitude |
| Residual-energy dispersion | 0.25 | variance of the high-pass residual |

**Grad-CAM / ResNet-101 / VGG-16 are gated off.** `GetFeatures.py` instantiates
`ResNet101()` and `VGG16()` at module scope — 180 MB + 553 MB of Keras weights.
That will OOM a 512 MB Render instance, so those stages are behind
`ENABLE_DEEP=1` and reported as unavailable otherwise. To enable them, move to a
paid plan and add `tensorflow-cpu` to `requirements.txt`.

---

## Run locally

Any Python 3.9+ with the five dependencies works; the project's own
`VideoForgeryCPU` env already has them.

```powershell
$E = "C:\Users\USER\anaconda3\envs\VideoForgeryCPU"
$env:PATH = "$E\Library\bin;$E;$E\Scripts;" + $env:PATH
cd webapp
& "$E\python.exe" -u app.py          # http://127.0.0.1:5000
```

Or clean-room:

```bash
pip install -r requirements.txt
python app.py
```

Smoke test against the clip the driver generates:

```bash
curl -s -X POST -F "video=@../driver_out/sample.mp4" \
     http://127.0.0.1:5000/api/analyze | head -c 400
```

## Deploy to Render

`render.yaml` is a blueprint — point Render at the repository and it picks it up:

```yaml
rootDir: webapp
buildCommand: pip install -r requirements.txt
startCommand: gunicorn --workers 1 --threads 2 --timeout 180 --preload app:app
healthCheckPath: /healthz
```

One worker, preloaded, 180 s timeout: video decoding is CPU-bound and a cold free
instance is slow to wake. `MAX_FRAMES` (default 48) bounds both latency and peak
memory — lower it if you see OOM restarts.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `PORT` | 5000 | bind port (Render sets this) |
| `MAX_FRAMES` | 48 | frames sampled per clip |
| `ENABLE_DEEP` | 0 | `1` enables the Keras stages (needs TensorFlow + >512 MB) |

## API

`POST /api/analyze` — multipart form, field `video`. Returns JSON:

```jsonc
{
  "meta":   { "sampled": 24, "total_frames": 30, "fps": 10.0,
              "face_roi": true, "elapsed_s": 0.26, "deep_features": false },
  "score":  { "value": 69.8, "band": "High", "terms": [ /* audit trail */ ] },
  "series": { "keyframe": [...], "flow": [...],
              "keyframe_peaks": [...], "flow_peaks": [...] },
  "glcm":   { "contrast": 1.568, "homogeneity": 0.683, ... },
  "stages": [ { "key": "roi", "title": "...", "image": "data:image/png;base64,..." } ]
}
```

`GET /healthz` → `{"status":"ok","deep_features":false,"max_frames":48}`

Result images are returned inline as base64 data URIs; nothing is written to
disk. The upload is deleted in the request's `finally` block.

## Verified

Exercised end to end against `driver_out/sample.mp4` (30 frames, 318 KB):
HTTP 200 in **0.26 s**, face ROI detected, all 10 stage images produced,
tamper-evidence index 69.8 ("High" — expected, since that clip is assembled from
unrelated frames and is genuinely discontinuous). `/` and `/about` both render;
screenshots taken headlessly against the running server.

## Layout

```
webapp/
  app.py                 Flask app + full pipeline
  requirements.txt       flask, gunicorn, numpy, scipy, opencv-python-headless
  render.yaml            Render blueprint
  Procfile               gunicorn entry (Heroku-style hosts)
  .python-version        3.12
  assets/                bundled Haar cascade
  templates/             index.html, about.html
  static/css/styles.css  dark forensic theme
  static/js/app.js       upload, fetch, render, inline SVG chart
```

`cv2.data` is missing from some conda OpenCV builds, so the cascade is bundled in
`assets/` and resolved with a fallback chain rather than assumed.
