"""Prepare real FaceForensics++ face crops with an identity-level split.

Turns DATASET/FFPP_C23 into a training set the published recipes expect:
face crops at 299x299, N frames per video, and - the part that matters most -
a split by source identity rather than by clip.

Why identity-level. A manipulated FF++ clip is named <target>_<source>.mp4 and
shares its underlying footage with original/<target>.mp4. Splitting by clip
puts near-duplicate footage on both sides of the boundary, and a detector then
scores highly by recognising the footage instead of the manipulation. Section
5.11.1 of the paper measures that effect on the small subset: for Xception RGB
frames, 57.64 % becomes 85.54 % balanced accuracy from the split alone (both
halves the same representation). FaceForensics++ defines a 720/140/140 partition
over the 1,000 identities precisely to prevent it, and this script reproduces
it: identity 0-719 train, 720-859 validation, 860-999 test. A manipulated clip
goes wherever its TARGET identity went, so no identity is ever split.

Output layout, ready for an ImageFolder-style loader:

    DATASET/ffpp_frames/{train,val,test}/{real,fake}/<video>_<frame>.jpg
    DATASET/ffpp_frames/manifest.csv     one row per crop
    DATASET/ffpp_frames/summary.json     counts and the exact split

Resumable: a video whose crops already exist is skipped.

    python Optimized/ffpp_prepare.py --methods Deepfakes --frames 32
    python Optimized/ffpp_prepare.py --methods Deepfakes,Face2Face,FaceSwap,NeuralTextures
"""
import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path(__file__).resolve().parents[1]
SRC = P / "DATASET" / "FFPP_C23" / "FaceForensics++_C23"
OUT = P / "DATASET" / "ffpp_frames"

# The FaceForensics++ split over the 1,000 source identities.
TRAIN_IDS = set(range(0, 720))
VAL_IDS = set(range(720, 860))
TEST_IDS = set(range(860, 1000))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def identity_of(stem, method):
    """Source identity that decides which partition a clip belongs to.

    original/123.mp4        -> 123
    Deepfakes/123_456.mp4   -> 123   (the target, whose footage it reuses)
    """
    if method == "original":
        return int(stem)
    return int(stem.split("_")[0])


def split_of(ident):
    if ident in TRAIN_IDS:
        return "train"
    if ident in VAL_IDS:
        return "val"
    return "test"


def face_crops(path, n_frames, size, margin, cascade):
    """Evenly spaced frames, largest detected face, margin, resized."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    idx = np.unique(np.linspace(0, max(total - 1, 0),
                                min(n_frames, total)).astype(int))

    # Seek to each wanted frame rather than decoding the whole clip. Reading
    # sequentially costs one decode per frame in the file - about 1,000 for a
    # median FF++ clip to collect 32 - and seeking cuts that by an order of
    # magnitude. Seeks on h264 land on the nearest keyframe, which is fine
    # here: the frames only need to be spread through the clip, not to sit at
    # exact indices.
    out, last = [], None
    for fi in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            last = (x, y, w, h)
        if last is None:
            continue
        x, y, w, h = last
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1 = min(frame.shape[1], x + w + mx)
        y1 = min(frame.shape[0], y + h + my)
        crop = frame[y0:y1, x0:x1]
        if crop.size:
            out.append((int(fi), cv2.resize(crop, (size, size))))
    cap.release()
    return out


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="Deepfakes",
                    help="comma-separated manipulation directories")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=299)
    ap.add_argument("--margin", type=float, default=0.25,
                    help="fraction of the box added on each side")
    ap.add_argument("--limit", type=int, default=0,
                    help="videos per class, 0 = all")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="run N copies with --shard 0..N-1 to use N cores; "
                         "each writes manifest_<shard>.csv, merged by "
                         "--merge")
    ap.add_argument("--merge", action="store_true",
                    help="combine manifest_*.csv into manifest.csv and exit")
    args = ap.parse_args()

    if args.merge:
        import pandas as pd
        parts = sorted(OUT.glob("manifest_*.csv"))
        if not parts:
            raise SystemExit(f"no manifest shards in {OUT}")
        df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
        df = df.drop_duplicates(subset="file").sort_values("file")
        df.to_csv(OUT / "manifest.csv", index=False)
        counts = df.groupby(["split", "label"]).size()
        summary = {"videos": int(df["video"].nunique()),
                   "crops": int(len(df)),
                   "counts": {f"{s}/{c}": int(v)
                              for (s, c), v in counts.items()},
                   "identity_level": True,
                   "shards_merged": [f.name for f in parts]}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
        log(f"merged {len(parts)} shards -> manifest.csv, {len(df)} crops")
        for k, v in summary["counts"].items():
            log(f"  {k:<14} {v:7d}")
        # No identity may straddle a partition; the whole protocol rests on it.
        ids = df.groupby("split")["identity"].apply(set)
        for a in ids.index:
            for b in ids.index:
                if a < b and (ids[a] & ids[b]):
                    raise SystemExit(f"identity leak {a}/{b}: "
                                     f"{sorted(ids[a] & ids[b])[:5]}")
        log("no identity appears in two splits")
        return

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    # This build of cv2 has no cv2.data (the same defect that broke the
    # webapp), so the cascade is located explicitly: the conda package's
    # haarcascades directory first, then the copy bundled in this repository.
    cand = [Path(sys.executable).parent / "Library" / "etc" / "haarcascades"
            / "haarcascade_frontalface_default.xml",
            P / "webapp" / "assets" / "haarcascade_frontalface_alt2.xml",
            P / "Temp" / "haarcascade_frontalface_alt2.xml"]
    cascade = None
    for c in cand:
        if c.exists():
            cascade = cv2.CascadeClassifier(str(c))
            if not cascade.empty():
                log(f"cascade: {c.name}")
                break
            cascade = None
    if cascade is None:
        raise SystemExit(f"no usable Haar cascade in: "
                         f"{[str(c) for c in cand]}")

    jobs = []
    for f in sorted((SRC / "original").glob("*.mp4")):
        jobs.append((f, "original", "real"))
    for m in methods:
        for f in sorted((SRC / m).glob("*.mp4")):
            jobs.append((f, m, "fake"))
    if args.limit:
        by = {}
        keep = []
        for j in jobs:
            k = (j[1],)
            by[k] = by.get(k, 0) + 1
            if by[k] <= args.limit:
                keep.append(j)
        jobs = keep

    # Interleaved rather than contiguous slicing, so every shard gets a mix of
    # 1080p originals and smaller manipulated clips and they finish together.
    if args.nshards > 1:
        jobs = jobs[args.shard::args.nshards]
    log(f"shard {args.shard + 1}/{args.nshards}: {len(jobs)} videos "
        f"(original + {', '.join(methods)})")
    log(f"{args.frames} frames each at {args.size}px, margin {args.margin}")
    log(f"split by identity: {len(TRAIN_IDS)} train / {len(VAL_IDS)} val / "
        f"{len(VAL_IDS)} test")

    rows, counts, t0, skipped = [], {}, time.time(), 0
    for n, (path, method, cls) in enumerate(jobs, 1):
        ident = identity_of(path.stem, method)
        sp = split_of(ident)
        d = OUT / sp / cls
        d.mkdir(parents=True, exist_ok=True)
        tag = f"{method}__{path.stem}"
        existing = sorted(d.glob(f"{tag}__f*.jpg"))
        if existing:
            skipped += 1
            for e in existing:
                # as_posix(), not str(): a backslash-separated path in the
                # manifest resolves to nothing on the Linux side (Colab), and
                # the failure only shows up after the archive is uploaded.
                rows.append({"file": e.relative_to(OUT).as_posix(), "split": sp,
                             "label": cls, "method": method,
                             "identity": ident, "video": path.stem})
            counts[(sp, cls)] = counts.get((sp, cls), 0) + len(existing)
        else:
            for fi, img in face_crops(path, args.frames, args.size,
                                      args.margin, cascade):
                fp = d / f"{tag}__f{fi:05d}.jpg"
                cv2.imwrite(str(fp), img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                rows.append({"file": fp.relative_to(OUT).as_posix(), "split": sp,
                             "label": cls, "method": method,
                             "identity": ident, "video": path.stem})
                counts[(sp, cls)] = counts.get((sp, cls), 0) + 1
        if n % 50 == 0 or n == len(jobs):
            el = time.time() - t0
            log(f"  {n}/{len(jobs)} videos  {len(rows)} crops  "
                f"{el / 60:.1f} min  eta {(el / n * (len(jobs) - n)) / 60:.1f} min")

    import pandas as pd
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    name = ("manifest.csv" if args.nshards == 1
            else f"manifest_{args.shard:02d}.csv")
    df.to_csv(OUT / name, index=False)

    summary = {
        "methods": methods, "frames_per_video": args.frames,
        "size": args.size, "margin": args.margin,
        "videos": len(jobs), "crops": len(df), "videos_skipped": skipped,
        "split": {"train": sorted(TRAIN_IDS)[:1] + ["..."] + [719],
                  "val": [720, "...", 859], "test": [860, "...", 999]},
        "counts": {f"{s}/{c}": int(v) for (s, c), v in sorted(counts.items())},
        "identity_level": True,
        "note": "a manipulated clip follows its TARGET identity, so footage "
                "never crosses the split boundary",
    }
    if args.nshards == 1:
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    log("\ncounts")
    for k, v in summary["counts"].items():
        log(f"  {k:<14} {v:7d}")
    log(f"\n{name}  {len(df)} rows")
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
