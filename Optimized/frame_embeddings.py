"""Per-frame backbone embeddings of the 'proposed' tensor.

The embeddings cached so far come from comparative1[:, :, :3] - a single
3-channel slice of one 128x128 feature map per video. That discards the time
axis entirely and 9 of the 12 channels of the richer tensor.

This builds the representation most likely to carry signal if any exists:

  * every frame embedded separately, so temporal structure survives into the
    aggregation rather than being averaged away beforehand;
  * all 12 channels covered, in 3-channel groups (a backbone takes 3 inputs);
  * frame-difference maps embedded as well, since face-swap and splice
    artefacts are inter-frame inconsistencies rather than per-frame anomalies;
  * aggregated over time by mean AND std - the std is the part that can
    express "this video is temporally inconsistent".

Output is cached as Optimized/cache/emb_<name>.npy, which optimize_v2.py picks
up automatically on its next run.
"""
import sys
import time
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKBONES = {
    "MobileNetV3Large": ("mobilenet_v3", "MobileNetV3Large", 224),
    "EfficientNetV2S": ("efficientnet_v2", "EfficientNetV2S", 224),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def to_rgb(block, size):
    """(H, W, 3) float -> backbone-ready [0, 255] at size x size."""
    import cv2
    x = cv2.resize(block.astype(np.float32), (size, size))
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max()) + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1) * 255.0


def main():
    import importlib
    import pickle

    with open(P / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    pr = np.asarray(data["proposed"], dtype=np.float32)   # (50,10,128,128,12)
    n, T, H, W, C = pr.shape
    log(f"proposed {pr.shape}")

    dif = np.abs(np.diff(pr, axis=1))                     # (50,9,H,W,12)
    triplets = [(0, 3), (3, 6), (6, 9), (9, 12)]
    cache = P / "Optimized" / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    for bname, (mod_name, cls_name, size) in BACKBONES.items():
        out_path = cache / f"emb_{bname}_frames.npy"
        if out_path.exists():
            log(f"{bname}: already cached")
            continue

        mod = importlib.import_module(f"keras.applications.{mod_name}")
        pre = getattr(mod, "preprocess_input", None)
        backbone = getattr(mod, cls_name)(weights="imagenet", include_top=False,
                                          input_shape=(size, size, 3),
                                          pooling="avg")
        backbone.trainable = False
        log(f"{bname}: {backbone.count_params()/1e6:.1f} M frozen params")

        feats = []
        for src, tag, nT in ((pr, "frames", T), (dif, "delta", T - 1)):
            for (a, b) in triplets:
                # one batch per video: nT frames of this channel triplet
                agg_mean, agg_std = [], []
                for i in range(n):
                    batch = np.stack([to_rgb(src[i, t, :, :, a:b], size)
                                      for t in range(nT)])
                    if pre is not None:
                        batch = pre(batch)
                    e = backbone.predict(batch, batch_size=16, verbose=0)
                    agg_mean.append(e.mean(0))
                    agg_std.append(e.std(0))
                feats.append(np.stack(agg_mean))
                feats.append(np.stack(agg_std))
                log(f"  {tag} ch{a}:{b} done "
                    f"({len(feats)}/{2*len(triplets)*2} blocks)")

        emb = np.concatenate(feats, axis=1).astype(np.float32)
        np.save(out_path, emb)
        log(f"{bname}: wrote {emb.shape} -> {out_path.name}")
        del backbone


if __name__ == "__main__":
    main()
