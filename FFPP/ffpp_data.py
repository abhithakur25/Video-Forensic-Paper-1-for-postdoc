"""FaceForensics++ ingestion: videos -> cached face crops, split by video.

Expected layout under --root (what the FF++ download script produces):

    original_sequences/youtube/c23/videos/*.mp4          label 0  (authentic)
    manipulated_sequences/FaceSwap/c23/videos/*.mp4      label 1  (forged)
    manipulated_sequences/Deepfakes/c23/videos/*.mp4     label 1
    manipulated_sequences/Face2Face/c23/videos/*.mp4     label 1
    manipulated_sequences/NeuralTextures/c23/videos/*.mp4label 1

THE ONE RULE THAT MATTERS
-------------------------
Splits are by VIDEO, never by frame. Frames from one video are near-duplicates
of each other; if any of them land in train while others land in test, the
model recognises the video rather than the manipulation and the reported
accuracy is meaningless. Most implausibly high deepfake numbers come from
exactly this mistake. Every function here carries video ids alongside the
frames so the split can be enforced and audited.

Manipulated FF++ filenames are "<target>_<source>.mp4" and the corresponding
original is "<target>.mp4". Those share the same underlying footage, so the
identity pairing is also honoured: a target identity never appears on both
sides of a split.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANIPULATIONS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_cascade():
    """Haar cascade, the same detector the research code uses. cv2.data is
    absent from some conda builds, so resolve through a fallback chain."""
    import cv2
    here = Path(__file__).resolve().parent
    cands = [here / "haarcascade_frontalface_alt2.xml",
             here.parent / "Temp" / "haarcascade_frontalface_alt2.xml",
             here.parent / "webapp" / "assets" / "haarcascade_frontalface_alt2.xml"]
    try:
        cands.insert(0, Path(cv2.data.haarcascades) /
                     "haarcascade_frontalface_alt2.xml")
    except AttributeError:
        pass
    for p in cands:
        if p.exists():
            c = cv2.CascadeClassifier(str(p))
            if not c.empty():
                return c
    raise SystemExit("no usable Haar cascade found")


def scan(root, manipulations, compression="c23"):
    """Return [(path, label, video_id, identity)]."""
    root = Path(root)
    items = []
    orig = root / "original_sequences" / "youtube" / compression / "videos"
    for p in sorted(orig.glob("*.mp4")):
        vid = p.stem                      # e.g. "033"
        items.append((p, 0, f"orig_{vid}", vid))
    for man in manipulations:
        d = root / "manipulated_sequences" / man / compression / "videos"
        for p in sorted(d.glob("*.mp4")):
            vid = p.stem                  # e.g. "033_097"
            target = vid.split("_")[0]    # identity that appears in the frame
            items.append((p, 1, f"{man}_{vid}", target))
    return items


def crop_faces(path, cascade, max_frames, size, margin=0.3,
               detector="haar"):
    """Sample frames uniformly and return face crops (n, size, size, 3) uint8."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return np.zeros((0, size, size, 3), np.uint8)
    idxs = np.unique(np.linspace(0, max(total - 1, 0), max_frames).astype(int))
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        if detector == "center":
            # square centre crop - used by the smoke test, where synthetic
            # frames contain no real faces for the cascade to find
            H, W = frame.shape[:2]
            side = min(H, W)
            x, y, w, h = (W - side) // 2, (H - side) // 2, side, side
            margin = 0.0
        else:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(grey, 1.1, 5, minSize=(60, 60))
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    cap.release()
    return (np.stack(out).astype(np.uint8) if out
            else np.zeros((0, size, size, 3), np.uint8))


def build_cache(root, out_dir, manipulations, compression, max_frames, size,
                limit=0, detector="haar"):
    import cv2  # noqa: F401  (import here so --help works without opencv)
    cascade = None if detector == "center" else find_cascade()
    items = scan(root, manipulations, compression)
    if not items:
        raise SystemExit(
            f"no videos under {root}. Expected e.g.\n"
            f"  {root}/original_sequences/youtube/{compression}/videos/*.mp4\n"
            f"  {root}/manipulated_sequences/FaceSwap/{compression}/videos/*.mp4")
    if limit:
        items = items[:limit]
    log(f"{len(items)} videos "
        f"({sum(1 for i in items if i[1]==0)} authentic / "
        f"{sum(1 for i in items if i[1]==1)} forged)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, labels, vids, idents, meta = [], [], [], [], []
    empty = 0
    for n, (path, label, vid, ident) in enumerate(items, 1):
        f = crop_faces(path, cascade, max_frames, size,
                       detector=detector)
        if len(f) == 0:
            empty += 1
        else:
            frames.append(f)
            labels.append(np.full(len(f), label, np.int8))
            vids.append(np.full(len(f), len(meta), np.int32))
            idents.append(np.full(len(f), len(meta), np.int32))
        meta.append({"video": str(path), "label": int(label), "id": vid,
                     "identity": ident, "frames": int(len(f))})
        if n % 25 == 0 or n == len(items):
            log(f"  {n}/{len(items)} videos, "
                f"{sum(len(x) for x in frames)} crops, {empty} with no face")

    X = np.concatenate(frames) if frames else np.zeros((0, size, size, 3), np.uint8)
    y = np.concatenate(labels) if labels else np.zeros((0,), np.int8)
    v = np.concatenate(vids) if vids else np.zeros((0,), np.int32)
    ident_names = sorted({m["identity"] for m in meta})
    imap = {a: i for i, a in enumerate(ident_names)}
    ident_of_video = np.array([imap[m["identity"]] for m in meta], np.int32)

    np.save(out_dir / "frames.npy", X)
    np.save(out_dir / "labels.npy", y)
    np.save(out_dir / "video_index.npy", v)
    np.save(out_dir / "identity_of_video.npy", ident_of_video)
    (out_dir / "meta.json").write_text(json.dumps({
        "videos": meta, "compression": compression,
        "manipulations": manipulations, "max_frames": max_frames,
        "crop_size": size,
        "n_videos": len(meta), "n_frames": int(len(X)),
        "videos_without_face": empty,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")
    log(f"cached {len(X)} face crops from {len(meta)} videos -> {out_dir}")
    log(f"  frames.npy {X.nbytes/1e9:.2f} GB")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="FaceForensics++ root")
    ap.add_argument("--out", default="FFPP/cache")
    ap.add_argument("--manipulations", default="FaceSwap",
                    help="comma separated, or 'all'")
    ap.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    ap.add_argument("--max-frames", type=int, default=32,
                    help="frames sampled per video")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0, help="first N videos (debug)")
    ap.add_argument("--detector", default="haar", choices=["haar", "center"],
                    help="'center' skips face detection; smoke tests only")
    a = ap.parse_args()
    mans = MANIPULATIONS if a.manipulations == "all" else \
        [m for m in a.manipulations.split(",") if m]
    build_cache(a.root, a.out, mans, a.compression, a.max_frames, a.size,
                a.limit, a.detector)


if __name__ == "__main__":
    main()
