"""Repack ffpp_frames for Drive with POSIX paths.

WHY THIS EXISTS
---------------
Two Windows-isms make the archive unusable on Colab, and both are silent:

1. .NET Framework's ZipFile.CreateFromDirectory writes entry names with
   backslashes. ZIP APPNOTE 4.4.17 requires forward slashes. Linux `unzip`
   does not treat '\' as a separator, so it creates 63,971 files with literal
   backslashes in their names in one flat directory - no train/val/test tree.
   The bug is fixed in .NET Core; PowerShell 5.1 runs on Framework.

2. ffpp_prepare.py built the manifest's `file` column with
   str(Path.relative_to()), which is backslash-separated on Windows, so
   os.path.join(ROOT, file) resolves to nothing on Linux even if the tree is
   correct.

Python's zipfile normalises arcnames to '/' itself, so building the archive
here fixes (1); the manifests are rewritten in place to fix (2).

    python Optimized/repack_ffpp.py
    python Optimized/repack_ffpp.py --verify-only
"""
import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path(__file__).resolve().parents[1]
OUT = P / "DATASET" / "ffpp_frames"
ZIP = P / "DATASET" / "ffpp_frames.zip"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fix_manifests():
    """Rewrite every manifest's `file` column to forward slashes."""
    changed = []
    for f in sorted(OUT.glob("manifest*.csv")):
        with io.open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        n = sum(1 for r in rows if "\\" in r["file"])
        if not n:
            log(f"  {f.name}: already POSIX")
            continue
        for r in rows:
            r["file"] = r["file"].replace("\\", "/")
        with io.open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        changed.append((f.name, n))
        log(f"  {f.name}: {n} paths -> POSIX")
    return changed


def build_zip():
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    log(f"packing {len(files):,} files (stored, no deflate)")
    t0 = time.time()
    tmp = ZIP.with_suffix(".zip.tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, p in enumerate(files, 1):
            # as_posix() on the relative path, and zipfile normalises further -
            # entry names are guaranteed '/'-separated.
            z.write(p, p.relative_to(OUT).as_posix())
            if i % 10000 == 0:
                log(f"  {i:,}/{len(files):,}")
    if ZIP.exists():
        ZIP.unlink()
    tmp.rename(ZIP)
    log(f"wrote {ZIP.name}  {ZIP.stat().st_size / 1e9:.2f} GB  "
        f"({(time.time() - t0) / 60:.1f} min)")


def verify():
    """Fail loudly on anything that would break the Colab side."""
    ok = True
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    log(f"entries: {len(names):,}")

    bad = [n for n in names if "\\" in n]
    log(f"entries containing a backslash: {len(bad)}"
        + (f"   e.g. {bad[0]}" if bad else "   OK"))
    ok &= not bad

    tops = sorted({n.split("/")[0] for n in names})
    log(f"top level: {tops}")
    ok &= {"train", "val", "test"}.issubset(set(tops))
    ok &= "manifest.csv" in tops

    with zipfile.ZipFile(ZIP) as z:
        with z.open("manifest.csv") as fh:
            rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))
    log(f"manifest rows: {len(rows):,}")
    mbad = [r["file"] for r in rows if "\\" in r["file"]]
    log(f"manifest paths with a backslash: {len(mbad)}"
        + (f"   e.g. {mbad[0]}" if mbad else "   OK"))
    ok &= not mbad

    # every manifest path must actually exist as an archive entry
    inzip = set(names)
    missing = [r["file"] for r in rows if r["file"] not in inzip]
    log(f"manifest rows with no matching entry: {len(missing)}"
        + (f"   e.g. {missing[0]}" if missing else "   OK"))
    ok &= not missing

    # the identity partition must still hold inside the shipped manifest
    ids = {}
    for r in rows:
        ids.setdefault(r["split"], set()).add(r["identity"])
    for a in ids:
        for b in ids:
            if a < b and (ids[a] & ids[b]):
                log(f"IDENTITY LEAK {a}/{b}: {sorted(ids[a] & ids[b])[:5]}")
                ok = False
    log("no identity appears in two splits" if ok else "LEAK CHECK FAILED")

    counts = {}
    for r in rows:
        counts[f"{r['split']}/{r['label']}"] = \
            counts.get(f"{r['split']}/{r['label']}", 0) + 1
    for k in sorted(counts):
        log(f"  {k:<12} {counts[k]:6,}")

    log("VERIFY " + ("PASSED" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        log("normalising manifest paths")
        fix_manifests()
        (OUT / "summary.json").write_text(
            json.dumps({**json.loads((OUT / "summary.json").read_text(
                encoding="utf-8")), "path_separator": "/"}, indent=2),
            encoding="utf-8")
        build_zip()
    raise SystemExit(0 if verify() else 1)


if __name__ == "__main__":
    main()
