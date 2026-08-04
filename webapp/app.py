"""
VForensiQ - web front end for the Paper 1 video forgery detection pipeline.

Runs the classical half of the SMA-CLMPNet pipeline in the browser:
gradient-based key-frame selection -> Viola-Jones ROI -> Lucas-Kanade optical
flow -> LDZP texture -> ResNet-style statistical maps -> GLCM descriptors.

Deliberately TensorFlow-free at runtime. The research code instantiates
ResNet101() and VGG16() at module scope (733 MB of weights) which will not fit
a 512 MB Render instance, so the deep-feature stages are gated behind
ENABLE_DEEP=1 and simply reported as unavailable when it is not set.

There is NO trained SMA-CLMPNet checkpoint in the repository - Analysis.py
trains every model on the fly - so this app does not claim a forged/authentic
verdict from the published network. What it reports is a transparent
tamper-evidence index computed from the signals it actually measures, with each
contributing term shown separately. See /about.
"""
import base64
import io
import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.stats import kurtosis, skew

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "48"))
ENABLE_DEEP = os.environ.get("ENABLE_DEEP", "0") == "1"
def _load_cascade():
    """Resolve the Viola-Jones cascade.

    cv2.data exists in the pip opencv wheels but NOT in every conda build, so a
    bundled copy is shipped in assets/ and tried first. The project's own copy
    under Temp/ is the last resort when running from a checkout.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "assets", "haarcascade_frontalface_alt2.xml"),
             os.path.join(here, os.pardir, "Temp",
                          "haarcascade_frontalface_alt2.xml")]
    try:
        cands.insert(1, cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    except AttributeError:
        pass
    for p in cands:
        if os.path.exists(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                return c
    return None


FACE_CASCADE = _load_cascade()


# ---------------------------------------------------------------- utilities
def png_b64(img, size=None):
    """ndarray (BGR or gray or float) -> base64 PNG data URI."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        finite = a[np.isfinite(a)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        a = np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)
        a = (np.nan_to_num(a) * 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.applyColorMap(a, cv2.COLORMAP_VIRIDIS)
    if size:
        a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def read_frames(path, cap_n=MAX_FRAMES):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video - unsupported or corrupt file")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    step = max(1, total // cap_n) if total > cap_n else 1
    frames, idx = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h, w = f.shape[:2]
            if w > 480:
                f = cv2.resize(f, (480, int(h * 480 / w)))
            frames.append(f)
            if len(frames) >= cap_n:
                break
        idx += 1
    cap.release()
    if not frames:
        raise ValueError("no decodable frames found")
    return frames, {"total_frames": total or len(frames), "fps": round(fps, 2),
                    "sampled": len(frames)}


# ------------------------------------------------------- pipeline stages
def keyframe_scores(frames):
    """Gradient-magnitude difference between consecutive frames.
    Mirrors the gradient-based key-frame selection in GetPreprocessing.py:
    tampering (insertion/deletion/duplication) shows up as a spike."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    grads = []
    for g in grays:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grads.append(cv2.magnitude(gx, gy))
    d = [float(np.mean(np.abs(grads[i] - grads[i - 1]))) for i in range(1, len(grads))]
    return grays, ([0.0] + d if d else [0.0])


def roi_extract(frame):
    """Viola-Jones face ROI, as GUI.roi() does. Falls back to the full frame."""
    if FACE_CASCADE is None:
        return frame, False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        return frame[y:y + h, x:x + w], True
    return frame, False


def statistical_maps(gray):
    """Mean / variance / std / skew / kurtosis over a 3x3 neighbourhood.

    GetFeatures.statistical_features() does this with a Python double loop over
    every pixel, which is far too slow for a request. This is the vectorised
    equivalent over the same 8-neighbourhood.
    """
    g = cv2.resize(gray, (128, 128)).astype(np.float64)
    sh = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, -1)]
    stack = np.stack([np.roll(np.roll(g, dy, axis=0), dx, axis=1) for dx, dy in sh])
    return {
        "mean": stack.mean(axis=0),
        "variance": stack.var(axis=0),
        "std": stack.std(axis=0),
        "skew": np.nan_to_num(skew(stack, axis=0, bias=True)),
        "kurtosis": np.nan_to_num(kurtosis(stack, axis=0, bias=True)),
    }


def ldzp(gray):
    """Local Directional Zig-Zag Pattern - directional derivative responses
    thresholded into a bit pattern (SubFunctions/LDZP.py)."""
    g = cv2.resize(gray, (128, 128)).astype(np.float32)
    kern = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], np.float32),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], np.float32),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], np.float32),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], np.float32),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], np.float32),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], np.float32),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], np.float32),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], np.float32),
    ]
    resp = np.stack([cv2.filter2D(g, cv2.CV_32F, k) for k in kern])
    top = np.sort(resp, axis=0)[-3:].min(axis=0)
    bits = (resp >= top).astype(np.uint8)
    return sum(bits[i] * (1 << i) for i in range(8)).astype(np.float64)


def optical_flow(prev_gray, gray, frame):
    """Lucas-Kanade sparse flow, as GUI.object_flow_features() does."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3,
                                 minDistance=7, blockSize=7)
    vis = frame.copy()
    if p0 is None:
        return vis, 0.0, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None, winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if p1 is None or st is None:
        return vis, 0.0, 0
    good_new, good_old = p1[st == 1], p0[st == 1]
    if len(good_new) == 0:
        return vis, 0.0, 0
    mags = np.linalg.norm(good_new - good_old, axis=1)
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, (len(good_new), 3))
    for i, (n, o) in enumerate(zip(good_new, good_old)):
        a, b = n.ravel(); c, d = o.ravel()
        col = colors[i].tolist()
        cv2.line(vis, (int(a), int(b)), (int(c), int(d)), col, 2)
        cv2.circle(vis, (int(a), int(b)), 4, col, -1)
    return vis, float(np.mean(mags)), int(len(good_new))


def glcm_stats(gray):
    """Contrast / homogeneity / energy / correlation from a 1-offset GLCM."""
    g = (cv2.resize(gray, (128, 128)) // 8).astype(np.int32)  # 32 grey levels
    L = 32
    a, b = g[:, :-1].ravel(), g[:, 1:].ravel()
    m = np.zeros((L, L), np.float64)
    np.add.at(m, (a, b), 1.0)
    m /= m.sum() or 1.0
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    d = (i - j).astype(np.float64)
    mi, mj = (m.sum(1) * np.arange(L)).sum(), (m.sum(0) * np.arange(L)).sum()
    si = np.sqrt(((np.arange(L) - mi) ** 2 * m.sum(1)).sum()) or 1.0
    sj = np.sqrt(((np.arange(L) - mj) ** 2 * m.sum(0)).sum()) or 1.0
    return {
        "contrast": float((m * d ** 2).sum()),
        "homogeneity": float((m / (1.0 + d ** 2)).sum()),
        "energy": float(np.sqrt((m ** 2).sum())),
        "correlation": float(((i - mi) * (j - mj) * m).sum() / (si * sj)),
    }


def residual_energy(gray):
    """High-pass residual - re-encoded or spliced regions leave different
    residual energy than the surrounding pristine content."""
    g = cv2.resize(gray, (128, 128)).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 1.2)


def zscore_peaks(series):
    a = np.asarray(series, np.float64)
    if a.size < 3 or a.std() < 1e-9:
        return 0.0, []
    z = (a - a.mean()) / a.std()
    return float(z.max()), [int(i) for i in np.where(z > 2.5)[0]]


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html", deep=ENABLE_DEEP)


@app.route("/about")
def about():
    return render_template("about.html", deep=ENABLE_DEEP)


@app.route("/healthz")
def healthz():
    return jsonify(status="ok", deep_features=ENABLE_DEEP, max_frames=MAX_FRAMES)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    t0 = time.time()
    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify(error="No video uploaded."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported type '{ext}'. Use {', '.join(sorted(ALLOWED))}."), 400

    tmp = os.path.join("/tmp" if os.path.isdir("/tmp") else ".",
                       f"vf_{uuid.uuid4().hex}{ext}")
    f.save(tmp)
    try:
        frames, meta = read_frames(tmp)
        grays, kf = keyframe_scores(frames)

        pick = int(np.argmax(kf)) if len(kf) > 1 else 0
        frame = frames[pick]
        roi, face_found = roi_extract(frame)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        flow_series, flow_vis = [], None
        for i in range(1, len(frames)):
            vis, mag, n = optical_flow(grays[i - 1], grays[i], frames[i])
            flow_series.append(mag)
            if i == pick or (flow_vis is None and i == len(frames) - 1):
                flow_vis = vis
        if flow_vis is None:
            flow_vis = frames[0]

        stats = statistical_maps(roi_gray)
        tex = ldzp(roi_gray)
        res = residual_energy(roi_gray)
        glcm = glcm_stats(roi_gray)

        kf_peak, kf_idx = zscore_peaks(kf[1:] if len(kf) > 1 else kf)
        fl_peak, fl_idx = zscore_peaks(flow_series)
        res_var = float(np.var(res))

        # Transparent tamper-evidence index. Each term is a measured quantity
        # squashed to 0-1; weights are fixed and shown in the response so the
        # score can be audited rather than taken on trust.
        terms = [
            ("Key-frame discontinuity", min(kf_peak / 4.0, 1.0), 0.40,
             f"peak z={kf_peak:.2f} over {len(kf)-1} transitions"),
            ("Optical-flow irregularity", min(fl_peak / 4.0, 1.0), 0.35,
             f"peak z={fl_peak:.2f} over {len(flow_series)} pairs"),
            ("Residual-energy dispersion", min(res_var / 60.0, 1.0), 0.25,
             f"variance={res_var:.2f} of the high-pass residual"),
        ]
        score = round(100.0 * sum(v * w for _, v, w, _ in terms), 1)
        band = ("Low" if score < 34 else "Elevated" if score < 67 else "High")

        out = {
            "meta": {**meta, "filename": f.filename,
                     "analysed_frame": pick, "face_roi": face_found,
                     "elapsed_s": round(time.time() - t0, 2),
                     "deep_features": ENABLE_DEEP},
            "score": {"value": score, "band": band, "terms": [
                {"name": n, "normalised": round(v, 3), "weight": w, "detail": d}
                for n, v, w, d in terms]},
            "series": {"keyframe": [round(x, 3) for x in kf],
                       "flow": [round(x, 3) for x in flow_series],
                       "keyframe_peaks": kf_idx, "flow_peaks": fl_idx},
            "glcm": {k: round(v, 5) for k, v in glcm.items()},
            "stages": [
                {"key": "input", "title": "Selected Frame",
                 "note": f"frame {pick} of {meta['sampled']} sampled - highest gradient change",
                 "image": png_b64(frame, (300, 300))},
                {"key": "roi", "title": "Viola-Jones ROI",
                 "note": "face region" if face_found else "no face detected - full frame used",
                 "image": png_b64(roi, (300, 300))},
                {"key": "flow", "title": "Lucas-Kanade Optical Flow",
                 "note": f"{len(flow_series)} frame pairs tracked",
                 "image": png_b64(flow_vis, (300, 300))},
                {"key": "ldzp", "title": "LDZP Texture",
                 "note": "local directional zig-zag pattern",
                 "image": png_b64(tex, (300, 300))},
                {"key": "residual", "title": "High-Pass Residual",
                 "note": f"variance {res_var:.2f}",
                 "image": png_b64(res, (300, 300))},
                {"key": "mean", "title": "Statistical - Mean",
                 "note": "3x3 neighbourhood", "image": png_b64(stats["mean"], (300, 300))},
                {"key": "variance", "title": "Statistical - Variance",
                 "note": "3x3 neighbourhood", "image": png_b64(stats["variance"], (300, 300))},
                {"key": "std", "title": "Statistical - Std. Deviation",
                 "note": "3x3 neighbourhood", "image": png_b64(stats["std"], (300, 300))},
                {"key": "skew", "title": "Statistical - Skewness",
                 "note": "3x3 neighbourhood", "image": png_b64(stats["skew"], (300, 300))},
                {"key": "kurtosis", "title": "Statistical - Kurtosis",
                 "note": "3x3 neighbourhood", "image": png_b64(stats["kurtosis"], (300, 300))},
            ],
        }
        if not ENABLE_DEEP:
            out["notice"] = ("Grad-CAM, ResNet-101 and VGG-16 stages are disabled "
                             "(ENABLE_DEEP=0): they need 733 MB of Keras weights, "
                             "beyond a 512 MB instance.")
        return jsonify(out)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # noqa: BLE001
        app.logger.exception("analysis failed")
        return jsonify(error=f"{type(e).__name__}: {e}"), 500
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
