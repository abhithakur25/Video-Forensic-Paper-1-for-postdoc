"""Remove every fabricated result from the working tree and the repository.

Fabricated artifacts are MOVED to a quarantine directory outside the repo,
not unlinked. The repository and working tree end up carrying only measured
results, which is what was asked for; the originals remain on disk one level
up in case they are ever needed to substantiate the integrity finding.

Three independent grounds for the classification, all verified:

  1. Scored through the tampered `mealpy/metrics.py`, which discards the
     model's predictions (`_check_targets`, lines 16-75).
  2. Plotted from (1) - every figure under Results/TP, Results/KF,
     Results/RocAnalysis.
  3. Describes a corpus that does not exist. Features.pkl holds 50 videos
     (29 authentic / 21 forged). Class.png claims 1000/1000;
     ConfusionMatrix.png totals 400 test samples at 200/200; Features.csv
     quotes accuracies to 12 significant figures. None of these can have
     been computed from the shipped data by any scorer.

Run with --apply to move; default is a dry run.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1]
QUARANTINE = P.parent / "_FABRICATED_QUARANTINE_Paper1"

# (path relative to repo root, why)
FABRICATED = [
    ("Analysis", "authors' own arrays, dated 2025-03-19; the source of every "
                 "metric figure in the manuscript, all via the tampered scorer"),
    ("Analysis1/TP", "training-percentage sweep re-run through the tampered scorer"),
    ("Analysis1/KF", "k-fold re-run through the tampered scorer"),
    ("Analysis1/TPR.npy", "ROC true-positive points from the invented vector"),
    ("Analysis1/FPR.npy", "ROC false-positive points from the invented vector"),
    ("Analysis1/TRUE_LATEST", "the run that exposed the tamper: four unrelated "
                              "backbones returning byte-identical scores"),
    ("Results/TP", "bar/line figures plotted from Analysis/TP"),
    ("Results/KF", "bar/line figures plotted from Analysis/KF"),
    ("Results/RocAnalysis", "ROC figures plotted from TPR.npy / FPR.npy"),
    ("Results/Results.xlsx", "the manuscript's TP and KF metric tables"),
    ("Results/Features.csv", "ablation accuracies 95.58-97.92, quoted to 12 "
                             "significant figures on a 50-video corpus"),
    ("Results/Features.jpg", "bar chart of Features.csv"),
    ("Results/ConfusionMatrix.png", "400 test samples at 200/200; the corpus "
                                    "is 50 videos at 29/21"),
    ("Results/Class.png", "claims 1000 Normal / 1000 Scam; the corpus is 29/21"),
    ("logs/evaluation_tp_sweep.log", "console record of the tampered sweep"),
    ("logs/evaluation_kfold.log", "console record of the tampered k-fold"),
    ("logs/evaluation_kfold_aborted.log", "console record of the aborted tampered k-fold"),
]

# Verified genuine, must survive. Listed so the intent is explicit and any
# future edit to FABRICATED that would swallow one of these is caught.
PROTECTED = [
    "Analysis1/TRUE",            # measured training-percentage sweep
    "Analysis1/TRUE_KF",         # measured k-fold
    "Results/ImageResults",      # real GradCAM/LDZP/flow/ResNet image outputs
    "Results/Arc.png",           # architecture diagram, not a result
    "Optimized", "FFPP", "Features", "SubFunctions", "webapp",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually move; default is a dry run")
    args = ap.parse_args()

    for prot in PROTECTED:
        for rel, _ in FABRICATED:
            if prot == rel or prot.startswith(rel + "/"):
                raise SystemExit(f"REFUSING: {rel!r} would remove protected "
                                 f"{prot!r}")

    moved, missing, total_files, total_bytes = [], [], 0, 0
    for rel, why in FABRICATED:
        src = P / rel
        if not src.exists():
            missing.append(rel)
            continue
        files = ([src] if src.is_file()
                 else [f for f in src.rglob("*") if f.is_file()])
        n, b = len(files), sum(f.stat().st_size for f in files)
        total_files += n
        total_bytes += b
        print(f"{'MOVE' if args.apply else 'would move'}  {rel:<38} "
              f"{n:>5} files  {b/1e6:>7.2f} MB")
        print(f"        {why}")
        if args.apply:
            dst = QUARANTINE / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(src), str(dst))
        moved.append({"path": rel, "reason": why, "files": n, "bytes": b})

    print(f"\n{total_files} files, {total_bytes/1e6:.2f} MB")
    if missing:
        print(f"absent already: {', '.join(missing)}")

    for prot in PROTECTED:
        p = P / prot
        print(f"  {'kept  ' if p.exists() else 'ABSENT'} {prot}")

    if not args.apply:
        print("\ndry run - nothing moved. Re-run with --apply.")
        return

    QUARANTINE.mkdir(parents=True, exist_ok=True)
    (QUARANTINE / "MANIFEST.json").write_text(json.dumps({
        "moved_from": str(P),
        "moved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "why": "fabricated results; see Optimized/INTEGRITY_FINDING.md",
        "entries": moved,
    }, indent=2), encoding="utf-8")
    (QUARANTINE / "README.md").write_text(
        "# Quarantined fabricated results (Paper 1)\n\n"
        "Moved out of the repository so it carries only measured results.\n"
        "Retained rather than deleted because they are the evidence for the\n"
        "integrity finding. **Nothing here is a measurement.**\n\n"
        "See `Implimentation_Paper1/Optimized/INTEGRITY_FINDING.md`.\n",
        encoding="utf-8")
    print(f"\nquarantine: {QUARANTINE}")

    r = subprocess.run(["git", "add", "-A"], cwd=str(P),
                       capture_output=True, text=True)
    print(f"git add -A -> {r.returncode} {r.stderr.strip()[:200]}")


if __name__ == "__main__":
    main()
