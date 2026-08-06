"""Write the outer-fold indices to disk so a torch-only process can reuse them.

torch and conda's MKL both ship an OpenMP runtime and abort when both load in
one process ("OMP: Error #15 ... libiomp5md.dll already initialized"). The
documented workaround, KMP_DUPLICATE_LIB_OK=TRUE, is explicitly described by
that message as liable to "silently produce incorrect results" - not a thing
to rely on in this project of all projects. So sklearn and torch never share a
process here: this script emits the folds, the torch script consumes them and
emits predictions, and a third script scores those predictions.

The split is byte-identical to the one in roc_confusion.py, so the STIL
comparison lands on exactly the same partitions as the existing baseline.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
SEED = 1234

with open(P / "Features" / "Features.pkl", "rb") as f:
    data = pickle.load(f)
y = np.asarray(data["labels"]).astype(int)

outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds = list(outer.split(np.zeros(len(y)), y))
np.savez(P / "Optimized" / "cache" / "folds.npz",
         **{f"tr{i}": tr for i, (tr, _) in enumerate(folds)},
         **{f"te{i}": te for i, (_, te) in enumerate(folds)},
         y=y, n_folds=np.array([len(folds)]))
for i, (tr, te) in enumerate(folds):
    print(f"fold {i}: {len(tr)} train / {len(te)} test  "
          f"test classes {np.bincount(y[te], minlength=2).tolist()}")
print(f"wrote Optimized/cache/folds.npz  (seed {SEED}, "
      f"StratifiedKFold(5, shuffle=True))")
