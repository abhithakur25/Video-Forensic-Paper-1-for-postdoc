"""Load the FaceForensics++ C23 metadata as a DataFrame.

This is the local equivalent of

    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
                                "xdxd003/ff-c23", file_path)

with two corrections. The snippet leaves file_path empty, which the pandas
adapter cannot resolve - it needs the path of one file inside the dataset, and
for this dataset that is csv/FF++_Metadata.csv. And the download is
unnecessary: archive.zip already extracted to DATASET/FFPP_C23 is the same
dataset, so pulling 16.7 GB again through a TLS-intercepting proxy would buy
nothing.

Set --kagglehub to take the remote path anyway; it needs kaggle.json present
and kagglehub >= 0.3.4, since load_dataset does not exist in 0.2.x.

    python Optimized/load_ffpp.py
    python Optimized/load_ffpp.py --file csv/original.csv
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = Path(__file__).resolve().parents[1]
ROOT = P / "DATASET" / "FFPP_C23" / "FaceForensics++_C23"
HANDLE = "xdxd003/ff-c23"
DEFAULT_FILE = "csv/FF++_Metadata.csv"


def load_local(file_path=DEFAULT_FILE):
    import pandas as pd
    f = ROOT / file_path
    if not f.exists():
        raise SystemExit(f"not found: {f}\nRun Optimized/extract_ffpp.py first.")
    return pd.read_csv(f)


def load_kagglehub(file_path=DEFAULT_FILE):
    import kagglehub
    if not hasattr(kagglehub, "load_dataset"):
        raise SystemExit(
            f"kagglehub {kagglehub.__version__} has no load_dataset; that "
            f"arrived in 0.3.4. Upgrade, or use the local path (default).")
    from kagglehub import KaggleDatasetAdapter
    return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, HANDLE,
                                  file_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--kagglehub", action="store_true",
                    help="fetch from Kaggle instead of the local extraction")
    args = ap.parse_args()

    df = (load_kagglehub(args.file) if args.kagglehub
          else load_local(args.file))

    print(f"source     {'kagglehub ' + HANDLE if args.kagglehub else ROOT}")
    print(f"file       {args.file}")
    print(f"shape      {df.shape}")
    print(f"columns    {list(df.columns)}\n")
    print("First 5 records:")
    print(df.head().to_string())

    if "Label" in df.columns:
        print(f"\nlabels\n{df['Label'].value_counts().to_string()}")
    if "File Path" in df.columns:
        meth = df["File Path"].str.split("/").str[0]
        print(f"\nmethod\n{meth.value_counts().to_string()}")
        real = meth.eq("original")
        print(f"\nreal {int(real.sum())} / fake {int((~real).sum())}")
    if "Frame Count" in df.columns:
        fc = df["Frame Count"]
        print(f"\nframes per video: min {fc.min()}  median {fc.median():.0f}  "
              f"max {fc.max()}  total {fc.sum():,}")
    if "Width" in df.columns:
        print("resolutions: "
              + ", ".join(f"{w}x{h} ({c})" for (w, h), c in
                          df.groupby(["Width", "Height"]).size()
                          .sort_values(ascending=False).head(5).items()))
    return df


if __name__ == "__main__":
    main()
