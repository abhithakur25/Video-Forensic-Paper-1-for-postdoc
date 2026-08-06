"""STIL's Temporal Inconsistency Module applied to this project's feature tensor.

Reference implementation: Tencent TFace, security/tasks/Face-Forgery-Detection/
STIL/models/ops.py (Gu et al., "Spatiotemporal Inconsistency Learning for
DeepFake Video Detection", ACM MM 2021). TIM_Module and ISM_Module are imported
from that file unmodified - see External/TFace, pinned by commit hash in the
manifest this script writes.

Why this model and not the published STIL network as shipped: STIL is an
SCNet-50 backbone consuming 8 RGB face crops at 224x224 and trained on the full
FaceForensics++ video set. Neither input is available here. This project has 50
videos already reduced to a (10, 128, 128, 12) tensor - 10 frames, 12 feature
channels, no RGB. The transferable part is the TIM block itself, which is what
the paper's ablation credits for most of its gain, and which is architecturally
independent of the backbone. It is dropped into a deliberately small stem sized
for 40 training samples.

Protocol: the same 5 outer folds as roc_confusion.py, read from
cache/folds.npz. Within each outer training fold an inner 20% stratified
validation split drives early stopping. Nothing is selected on the outer test
fold. Out-of-fold probabilities are written to disk for a separate process to
score, because torch and conda's MKL cannot share a process here.

Run in a torch-only process: no sklearn, no scipy, no skimage.
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

P = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
STIL = (P / "External" / "TFace" / "security" / "tasks" /
        "Face-Forgery-Detection" / "STIL")
sys.path.insert(0, str(STIL))
from models.ops import ISM_Module, TIM_Module  # noqa: E402

SEED = 1234


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class STILNet(nn.Module):
    """Small stem + the paper's TIM/ISM blocks + temporal average pooling.

    Channel widths are kept low on purpose. With 40 training videos the
    binding constraint is sample count, not capacity; a wide network here
    would only fit noise faster.
    """

    def __init__(self, in_ch=12, n_segment=10, width=32, n_class=2,
                 dropout=0.5):
        super().__init__()
        self.n_segment = n_segment
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 128 -> 32
        )
        self.tim1 = TIM_Module(w, reduction=4, n_segment=n_segment)
        self.ism1 = ISM_Module()
        self.mid = nn.Sequential(
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w * 2), nn.ReLU(inplace=True),   # 32 -> 16
        )
        self.tim2 = TIM_Module(w * 2, reduction=8, n_segment=n_segment)
        self.ism2 = ISM_Module()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(w * 2, n_class)

    def forward(self, x):
        """x: [n, t, c, h, w]"""
        n, t = x.shape[:2]
        x = x.reshape(n * t, *x.shape[2:])
        x = self.stem(x)
        x = self.ism1(self.tim1(x))
        x = self.mid(x)
        x = self.ism2(self.tim2(x))
        x = self.pool(x).flatten(1)                 # [nt, 2w]
        x = x.view(n, t, -1).mean(1)                # temporal average
        return self.fc(self.drop(x))


def balanced_acc(y, p):
    out = []
    for c in (0, 1):
        m = y == c
        out.append(float((p[m] == c).mean()) if m.any() else 0.0)
    return float(np.mean(out))


def inner_split(y_tr, frac=0.2, seed=SEED):
    """Stratified inner validation split, numpy only."""
    rng = np.random.default_rng(seed)
    va = []
    for c in (0, 1):
        idx = np.where(y_tr == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac)))
        va.extend(idx[:k].tolist())
    va = np.array(sorted(va))
    tr = np.array([i for i in range(len(y_tr)) if i not in set(va.tolist())])
    return tr, va


def run_fold(Xtr, ytr, Xte, args, fold):
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    itr, iva = inner_split(ytr)
    xt = torch.from_numpy(Xtr[itr]).float()
    yt = torch.from_numpy(ytr[itr]).long()
    xv = torch.from_numpy(Xtr[iva]).float()
    yv = ytr[iva]

    model = STILNet(in_ch=Xtr.shape[2], n_segment=Xtr.shape[1],
                    width=args.width, dropout=args.dropout)
    cnt = np.bincount(ytr[itr], minlength=2).astype(np.float64)
    cw = torch.from_numpy((cnt.sum() / (2 * np.maximum(cnt, 1)))).float()
    lossf = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xv).argmax(1).numpy()
        bva = balanced_acc(yv, pv)
        if bva > best:
            best, best_ep = bva, ep
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            log(f"    fold {fold} ep {ep+1:3d}  loss {tot/len(xt):.4f}  "
                f"inner-val bal {bva*100:6.2f}  (best {best*100:.2f} @ep{best_ep+1})")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte).float())
        prob = torch.softmax(logits, 1)[:, 1].numpy()
    return prob, (prob >= 0.5).astype(int), best, best_ep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=6,
                    help="cap so a concurrent k-fold run is not starved")
    ap.add_argument("--tag", default="stil_tim")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    t0 = time.time()

    with open(P / "Features" / "Features.pkl", "rb") as f:
        data = pickle.load(f)
    pr = np.asarray(data["proposed"], dtype=np.float32)   # (50,10,128,128,12)
    X = np.transpose(pr, (0, 1, 4, 2, 3))                 # -> (n,t,c,h,w)
    y = np.asarray(data["labels"]).astype(int)
    log(f"X {X.shape}  y {np.bincount(y).tolist()}")

    # standardise per channel on the whole corpus: this uses no labels, so it
    # leaks nothing about the split. Per-fold statistics were also tried and
    # changed the result by less than a point.
    mu = X.mean((0, 1, 3, 4), keepdims=True)
    sd = X.std((0, 1, 3, 4), keepdims=True) + 1e-6
    X = (X - mu) / sd

    f = np.load(P / "Optimized" / "cache" / "folds.npz")
    n_folds = int(f["n_folds"][0])
    prob = np.full(len(y), np.nan)
    pred = np.full(len(y), -1, int)
    per_fold = []
    for i in range(n_folds):
        tr, te = f[f"tr{i}"], f[f"te{i}"]
        log(f"fold {i}: {len(tr)} train / {len(te)} test "
            f"(test classes {np.bincount(y[te], minlength=2).tolist()})")
        p, q, bva, bep = run_fold(X[tr], y[tr], X[te], args, i)
        prob[te], pred[te] = p, q
        fb = balanced_acc(y[te], q)
        per_fold.append({"fold": i, "inner_val_bal": bva,
                         "best_epoch": int(bep + 1), "outer_bal": fb})
        log(f"  fold {i} outer-test bal {fb*100:6.2f} "
            f"(inner-val {bva*100:.2f} @ep{bep+1})")

    assert not np.isnan(prob).any()
    pooled = balanced_acc(y, pred)
    mean_fold = float(np.mean([d["outer_bal"] for d in per_fold]))
    log(f"pooled out-of-fold balanced accuracy {pooled*100:.2f}")
    log(f"mean of per-fold balanced accuracies  {mean_fold*100:.2f}")

    try:
        commit = subprocess.run(
            ["git", "-C", str(P / "External" / "TFace"), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"

    out = P / "Optimized" / f"oof_{args.tag}.npz"
    np.savez(out, prob=prob, pred=pred, y=y)
    (P / "Optimized" / f"oof_{args.tag}.json").write_text(json.dumps({
        "model": "STIL TIM+ISM blocks on the proposed tensor",
        "reference": "Tencent/TFace security/tasks/Face-Forgery-Detection/STIL",
        "tface_commit": commit,
        "modules_imported_unmodified": ["TIM_Module", "ISM_Module"],
        "input": list(X.shape), "protocol":
            "5 outer folds from cache/folds.npz; inner 20% stratified "
            "validation split for early stopping; nothing selected on the "
            "outer test fold",
        "hyperparameters": vars(args),
        "pooled_balanced_accuracy": pooled,
        "mean_fold_balanced_accuracy": mean_fold,
        "per_fold": per_fold,
        "wall_seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    log(f"wrote {out.name} and oof_{args.tag}.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
