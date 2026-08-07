"""Regenerate everything downstream of the k-fold run, in dependency order.

Run once the k-fold reaches k = 10. Refuses to run before that: a half-finished
sweep silently regenerating the figures and both documents is how a stale
number ends up in a published table, which is the failure this project exists
to correct.

Order matters. Figures read the arrays; the documents embed the figures; the
log index describes the finished log.

    python Optimized/finalize_kfold.py            # verify, then regenerate
    python Optimized/finalize_kfold.py --check    # verify only
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
KF = P / "Analysis1" / "TRUE_KF"
EXPECTED_KS = [6, 7, 8, 9, 10]

STAGES = [
    ("figures", "Optimized/make_figures.py"),
    ("comparison chart", "Optimized/make_comparison_figure.py"),
    ("RESULTS.md", "Optimized/report.py"),
    ("work report", "Optimized/make_report_doc.py"),
    ("research article", "Optimized/make_article.py"),
    ("log index", "Optimized/save_logs.py"),
]


def verify():
    """Every model must carry one row per k, and the manifest must agree."""
    man = json.loads((KF / "run_manifest.json").read_text("utf-8"))
    ks = [int(k) for k in man["k_values"]]
    problems = []
    if ks != EXPECTED_KS:
        problems.append(f"manifest k_values = {ks}, expected {EXPECTED_KS}")
    for m in man["models"]:
        f = KF / f"{m}.npy"
        if not f.exists():
            problems.append(f"{m}.npy missing")
            continue
        a = np.load(f)
        if a.shape != (len(EXPECTED_KS), 6):
            problems.append(f"{m}.npy has shape {a.shape}, "
                            f"expected ({len(EXPECTED_KS)}, 6)")
    log = P / "logs" / "kfold_true.log"
    txt = log.read_text("utf-8", "replace")
    if "k values done: [6, 7, 8, 9, 10]" not in txt:
        problems.append("log has no checkpoint line for the full k sweep")
    print(f"manifest k_values : {ks}")
    print(f"models            : {len(man['models'])}")
    print(f"log               : {log.stat().st_size / 1e6:.2f} MB")
    for p in problems:
        print(f"  PROBLEM: {p}")
    return problems


def run(label, script):
    print(f"\n{'=' * 70}\n{label}  ({script})\n{'=' * 70}")
    env_py = Path(sys.executable)
    r = subprocess.run([str(env_py), str(P / script)], cwd=str(P))
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify only, regenerate nothing")
    args = ap.parse_args()

    problems = verify()
    if problems:
        raise SystemExit("\nnot finalising: the k-fold run is incomplete or "
                         "inconsistent. Fix the problems above first.")
    print("\nk-fold is complete and consistent.")
    if args.check:
        return
    for label, script in STAGES:
        run(label, script)
    print(f"\n{'=' * 70}\nall regenerated. Review, then commit "
          f"Analysis1/TRUE_KF/, the figures, both .docx and the log index.")


if __name__ == "__main__":
    main()
