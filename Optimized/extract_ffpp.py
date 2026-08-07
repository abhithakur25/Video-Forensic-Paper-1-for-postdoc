"""Unpack the FaceForensics++ C23 archive into DATASET/.

7,000 videos, 17.9 GB uncompressed: 1,000 pristine originals and 1,000 each
from six manipulation methods (DeepFakeDetection, Deepfakes, Face2Face,
FaceShifter, FaceSwap, NeuralTextures), plus per-method CSVs and a metadata
table.

This is the corpus the repository's frame-level pipeline was written for and
has never had. Everything measured so far in this project used a cached
50-video subset, and the ceiling established there - 74 % accuracy from an
out-of-fold AUC of 0.73 - is a statement about that subset, not about the
method. With the real corpus the identity-level split that FaceForensics++
defines becomes possible, which is the protocol every published 95 %+ figure
is measured under.

Extraction is resumable: a member already present at the right size is
skipped, so an interrupted run costs nothing.

    python Optimized/extract_ffpp.py --check      # space and plan only
    python Optimized/extract_ffpp.py
"""
import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path(__file__).resolve().parents[1]
ARCHIVE = P.parent / "archive.zip"
DEST = P / "DATASET" / "FFPP_C23"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args()
    dest = Path(args.dest)

    if not ARCHIVE.exists():
        raise SystemExit(f"archive not found: {ARCHIVE}")
    z = zipfile.ZipFile(ARCHIVE)
    info = z.infolist()
    total = sum(i.file_size for i in info)
    free = shutil.disk_usage(dest.anchor).free

    print(f"archive   {ARCHIVE}  ({human(ARCHIVE.stat().st_size)})")
    print(f"members   {len(info)}")
    print(f"expands   {human(total)}")
    print(f"dest      {dest}")
    print(f"free      {human(free)}  ->  {human(free - total)} after")
    if free < total * 1.05:
        raise SystemExit("not enough free space (want 5% headroom)")
    if args.check:
        return

    dest.mkdir(parents=True, exist_ok=True)
    done_bytes, skipped, written = 0, 0, 0
    t0 = time.time()
    for n, i in enumerate(info, 1):
        out = dest / i.filename
        if out.exists() and out.stat().st_size == i.file_size:
            skipped += 1
            done_bytes += i.file_size
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            written += 1
            done_bytes += i.file_size
        if n % 100 == 0 or n == len(info):
            el = time.time() - t0
            frac = done_bytes / total
            eta = (el / frac - el) if frac > 0.001 else 0
            print(f"[{time.strftime('%H:%M:%S')}] {n:5d}/{len(info)}  "
                  f"{frac * 100:5.1f}%  {human(done_bytes)}  "
                  f"{done_bytes / 1e6 / max(el, 1):.0f} MB/s  "
                  f"eta {eta / 60:.1f} min", flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: "
          f"{written} written, {skipped} already present")
    mp4 = sum(1 for _ in dest.rglob("*.mp4"))
    csv = sum(1 for _ in dest.rglob("*.csv"))
    print(f"on disk: {mp4} mp4, {csv} csv")


if __name__ == "__main__":
    main()
