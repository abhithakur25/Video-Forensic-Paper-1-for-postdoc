"""End-to-end smoke test with synthetic videos in FaceForensics++ layout.

Verifies the whole pipeline before the real dataset arrives:
  1. builds a fake FF++ tree with the exact directory and filename convention;
  2. runs ingestion (centre-crop mode, since synthetic frames have no faces);
  3. runs training in every mode;
  4. asserts the leakage guard actually fires when identities are shared.

The synthetic "forged" clips carry a faint periodic flicker the authentic ones
lack, so a working pipeline should score above chance. This checks the
plumbing, not detection quality.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / "FFPP"))
PY = sys.executable


def make_video(path, n_frames=24, size=160, forged=False, seed=0):
    import cv2
    rs = np.random.RandomState(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                         (size, size))
    base = rs.randint(60, 190, (size, size, 3)).astype(np.float32)
    base = cv2.GaussianBlur(base, (9, 9), 0)
    for t in range(n_frames):
        f = base + rs.normal(0, 4, base.shape)
        if forged:
            # periodic inter-frame discontinuity - the artefact a real
            # face-swap leaves and what the detector should key on
            f[40:120, 40:120] += 18.0 * ((t % 3) == 0)
        vw.write(np.clip(f, 0, 255).astype(np.uint8))
    vw.release()


def build_tree(root, n_ident=12):
    orig = root / "original_sequences" / "youtube" / "c23" / "videos"
    man = root / "manipulated_sequences" / "FaceSwap" / "c23" / "videos"
    for i in range(n_ident):
        make_video(orig / f"{i:03d}.mp4", forged=False, seed=i)
    for i in range(n_ident):
        j = (i + 1) % n_ident
        make_video(man / f"{i:03d}_{j:03d}.mp4", forged=True, seed=100 + i)
    return root


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd[1:]), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       cwd=str(P), **kw)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd[1:])}")
    return r.stdout


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ffpp_smoke_"))
    cache = tmp / "cache"
    try:
        print("1. building synthetic FF++ tree")
        build_tree(tmp / "data")
        n = len(list((tmp / "data").rglob("*.mp4")))
        print(f"   {n} videos")

        print("2. ingestion")
        out = run([PY, "FFPP/ffpp_data.py", "--root", tmp / "data",
                   "--out", cache, "--manipulations", "FaceSwap",
                   "--max-frames", 8, "--size", 96, "--detector", "center"])
        print("  ", out.strip().splitlines()[-2])
        meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
        assert meta["n_videos"] == n, meta["n_videos"]
        assert meta["n_frames"] > 0
        print(f"   cached {meta['n_frames']} crops from {meta['n_videos']} videos")

        print("3. leakage guard")
        sys.path.insert(0, str(P / "Optimized"))
        from ffpp_train import assert_disjoint
        g = np.array([0, 0, 1, 1, 2, 2])
        try:
            assert_disjoint(g, np.array([0, 1, 2]), np.array([2, 3]), "bad")
            raise SystemExit("FAILED: leakage guard did not fire")
        except SystemExit as e:
            if "LEAKAGE" not in str(e):
                raise
        assert_disjoint(g, np.array([0, 1]), np.array([4, 5]), "good")
        print("   guard fires on shared identities, passes on disjoint")

        print("4. training (single split, frozen backbone)")
        out = run([PY, "FFPP/ffpp_train.py", "--cache", cache,
                   "--backbone", "MobileNetV3Large", "--mode", "single",
                   "--epochs", 2, "--batch", 8, "--size", 96, "--freeze",
                   "--out", tmp / "results"])
        for line in out.strip().splitlines():
            if "level" in line or "integrity" in line:
                print("  ", line.strip())

        print("5. training-percentage mode")
        out = run([PY, "FFPP/ffpp_train.py", "--cache", cache,
                   "--backbone", "MobileNetV3Large", "--mode", "tp",
                   "--epochs", 1, "--batch", 8, "--size", 96, "--freeze",
                   "--out", tmp / "results"])
        tail = [l for l in out.strip().splitlines() if "%" in l][-7:]
        for line in tail:
            print("  ", line.strip())

        print("6. k-fold mode")
        run([PY, "FFPP/ffpp_train.py", "--cache", cache,
             "--backbone", "MobileNetV3Large", "--mode", "kfold", "--folds", 3,
             "--epochs", 1, "--batch", 8, "--size", 96, "--freeze",
             "--out", tmp / "results"])
        res = list((tmp / "results").glob("*.json"))
        print(f"   wrote {len(res)} result files")

        print()
        print("SMOKE TEST PASSED - pipeline is ready for real data")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
