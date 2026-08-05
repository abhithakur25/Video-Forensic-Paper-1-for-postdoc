"""One command: preflight -> ingest -> frozen baseline -> report.

Run this once the FaceForensics++ videos are in DATASET/. It checks the tree,
caches face crops, trains the frozen-backbone baseline at a single 80/20 split
and across the six training percentages, and writes FFPP/BASELINE_REPORT.md.

Frozen backbone (ImageNet features + trained head) rather than a full
fine-tune, because this machine is CPU-only: frozen is hours, fine-tuning is
days. It is a legitimate result in its own right and the natural first
baseline; --finetune switches to the full version where a GPU exists.

Safe to re-run: ingestion is skipped if the cache is already present and
matches the requested settings, so an interrupted run resumes cheaply.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = sys.executable


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def preflight(root, compression, manipulations):
    root = Path(root)
    orig = root / "original_sequences" / "youtube" / compression / "videos"
    n_orig = len(list(orig.glob("*.mp4"))) if orig.exists() else 0
    counts = {"original": n_orig}
    total = n_orig
    for m in manipulations:
        d = root / "manipulated_sequences" / m / compression / "videos"
        c = len(list(d.glob("*.mp4"))) if d.exists() else 0
        counts[m] = c
        total += c

    log("preflight")
    for k, v in counts.items():
        log(f"    {k:<16} {v:>5} videos")
    if total == 0:
        raise SystemExit(
            f"\nNo videos found under {root}.\n"
            f"Expected e.g. {orig}\\*.mp4\n"
            f"Download with the FF++ script:\n"
            f"  python faceforensics_download_v4.py {root} "
            f"-d original -c {compression} -t videos\n"
            f"  python faceforensics_download_v4.py {root} "
            f"-d FaceSwap -c {compression} -t videos")
    if n_orig == 0:
        raise SystemExit("originals missing - only manipulated videos found")
    if total - n_orig == 0:
        raise SystemExit("no manipulated videos found")

    free = shutil.disk_usage(str(P)).free / 1e9
    log(f"    free disk       {free:>5.1f} GB")
    return counts, total


def cache_ok(cache, max_frames, size, manipulations):
    m = Path(cache) / "meta.json"
    if not m.exists():
        return False
    try:
        j = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (j.get("max_frames") == max_frames and j.get("crop_size") == size
            and set(j.get("manipulations", [])) == set(manipulations)
            and (Path(cache) / "frames.npy").exists())


def run(cmd, tee=None):
    log("$ " + " ".join(str(c) for c in cmd[1:]))
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(P),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        print("   " + line.rstrip(), flush=True)
    proc.wait()
    if tee:
        Path(tee).write_text("\n".join(lines), encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"step failed (exit {proc.returncode})")
    return lines


def table(rows, head):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i])
                                       for i, c in enumerate(r)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
                     + [line(r) for r in rows])


def report(counts, meta, args, results_dir):
    COLS = ["ACC", "SEN", "SPE", "PRE", "F1", "BAL"]
    out = ["# FaceForensics++ baseline report\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}.\n",
           f"Backbone **{args.backbone}**, "
           f"{'fine-tuned' if args.finetune else 'frozen (ImageNet features + trained head)'}"
           f", {args.epochs} epochs, batch {args.batch}, crop {args.size}px, "
           f"{args.max_frames} frames/video, compression {args.compression}.\n",
           "Splits are grouped by source identity; the run halts on any "
           "shared identity between train and test. Scored with "
           "`Optimized/metrics_fixed.py`.\n",
           "\n## Corpus\n",
           table([[k, str(v)] for k, v in counts.items()]
                 + [["**face crops**", str(meta["n_frames"])],
                    ["videos with no face", str(meta["videos_without_face"])]],
                 ["Source", "Count"])]

    for f in sorted(Path(results_dir).glob(f"{args.backbone}_*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        out.append(f"\n## Mode: {j['mode']}\n")
        rows = []
        for k, r in j["results"].items():
            rows.append([k] + [f"{x*100:.2f}" for x in r["video"]]
                        + [f"{r['frame'][0]*100:.2f}", f"{r['frame'][5]*100:.2f}"])
        if rows:
            vids = np.array([r["video"] for r in j["results"].values()])
            rows.append(["**Mean**"] + [f"{v*100:.2f}" for v in vids.mean(0)]
                        + ["", ""])
        out.append(table(rows, ["Split"] + [f"video {c}" for c in COLS]
                         + ["frame ACC", "frame BAL"]))

    out.append("\n## Interpretation\n")
    out.append("Low-to-mid 90s video-level accuracy is consistent with "
               "published FF++ c23 results. **Near 99-100% should be treated "
               "as a leakage bug rather than a result** - check the `split "
               "integrity OK` lines in the log and confirm identity grouping "
               "in `meta.json` before reporting it.\n")
    out.append("These numbers are not comparable with the 50-video feature "
               "evaluation in `Optimized/RESULTS.md`, which has 19-44 "
               "training samples. That is the point of this pipeline.\n")
    p = P / "FFPP" / "BASELINE_REPORT.md"
    p.write_text("\n".join(out), encoding="utf-8")
    log(f"wrote {p.relative_to(P)}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="DATASET")
    ap.add_argument("--cache", default="FFPP/cache")
    ap.add_argument("--out", default="FFPP/results")
    ap.add_argument("--backbone", default="EfficientNetV2B0")
    ap.add_argument("--manipulations", default="FaceSwap")
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--finetune", action="store_true",
                    help="full fine-tune instead of frozen (needs a GPU)")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N videos - use for a quick trial")
    ap.add_argument("--detector", default="haar", choices=["haar", "center"],
                    help="'center' skips face detection; smoke tests only")
    ap.add_argument("--skip-tp", action="store_true",
                    help="single split only, skip the 6-point sweep")
    args = ap.parse_args()
    mans = [m for m in args.manipulations.split(",") if m]

    t0 = time.time()
    counts, total = preflight(args.root, args.compression, mans)

    if cache_ok(args.cache, args.max_frames, args.size, mans) and not args.limit:
        log(f"cache present and matching - skipping ingestion "
            f"({args.cache})")
    else:
        cmd = [PY, "-u", "FFPP/ffpp_data.py", "--root", args.root,
               "--out", args.cache, "--manipulations", args.manipulations,
               "--compression", args.compression,
               "--max-frames", args.max_frames, "--size", args.size,
               "--detector", args.detector]
        if args.limit:
            cmd += ["--limit", args.limit]
        run(cmd)

    meta = json.loads((Path(args.cache) / "meta.json").read_text("utf-8"))
    log(f"cache: {meta['n_frames']} crops from {meta['n_videos']} videos")

    # A silent zero here would train on an empty set and report nonsense.
    if meta["n_frames"] == 0:
        raise SystemExit(
            "\nNo face crops were extracted from any video.\n"
            "  - videos decoded but no face detected: check the Haar cascade "
            "resolves (FFPP/ffpp_data.py find_cascade)\n"
            "  - or the videos failed to decode: check OpenCV can read one "
            "with cv2.VideoCapture\n"
            "Nothing was trained.")
    miss = meta["videos_without_face"] / max(1, meta["n_videos"])
    if miss > 0.25:
        log(f"    WARNING: {miss:.0%} of videos yielded no face. Detection is "
            f"failing on a large fraction of the corpus; results will be "
            f"biased toward whatever it does detect.")
    per_video = meta["n_frames"] / max(1, meta["n_videos"] - meta["videos_without_face"])
    log(f"    {per_video:.1f} crops per usable video")

    common = [PY, "-u", "FFPP/ffpp_train.py", "--cache", args.cache,
              "--backbone", args.backbone, "--epochs", args.epochs,
              "--batch", args.batch, "--size", args.size, "--out", args.out]
    if not args.finetune:
        common += ["--freeze"]

    log("=== single 80/20 split")
    run(common + ["--mode", "single"], tee=P / "logs" / "ffpp_single.log")

    if not args.skip_tp:
        log("=== training-percentage sweep")
        run(common + ["--mode", "tp"], tee=P / "logs" / "ffpp_tp.log")

    txt = report(counts, meta, args, P / args.out)
    log(f"done in {time.time()-t0:.0f}s")
    print()
    print(txt)


if __name__ == "__main__":
    main()
