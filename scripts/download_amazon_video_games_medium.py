"""
Download and prepare the Amazon Reviews 2023 Video Games (medium subset)
for HyDART-RQ reverse top-k query experiments.

Source  : McAuley-Lab/Amazon-Reviews-2023 on Hugging Face
Method  : Direct file access via huggingface_hub.HfFileSystem
          (avoids load_dataset() which no longer supports dataset scripts)
Output  : data/raw/amazon_video_games/amazon_video_games_medium.csv
          data/raw/amazon_video_games/dataset_summary.txt
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks (fail fast with a clear message)
# ---------------------------------------------------------------------------
def _require(pkg, install_hint):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"ERROR: '{pkg}' is not installed.")
        print(f"       Fix : {install_hint}")
        sys.exit(1)

_require("huggingface_hub", "pip install huggingface_hub")
_require("pandas",          "pip install pandas")
_require("pyarrow",         "pip install pyarrow")
_require("tqdm",            "pip install tqdm")

import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfFileSystem

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO          = "McAuley-Lab/Amazon-Reviews-2023"
REPO_HF_PATH  = f"datasets/{REPO}"

# Ordered candidate paths - the script picks the first one that exists.
# The benchmark/5core/rating_only CSV is already 5-core filtered (47 MB),
# making it the ideal starting point for a medium subset.
CANDIDATE_PATHS = [
    # Best: 5-core filtered, rating-only (user_id, parent_asin, rating, timestamp)
    f"{REPO_HF_PATH}/benchmark/5core/rating_only/Video_Games.csv",
    # Fallback: 0-core (all interactions, 264 MB - larger but still manageable)
    f"{REPO_HF_PATH}/benchmark/0core/rating_only/Video_Games.csv",
    # Last resort: full raw JSONL (2.7 GB - slow but complete)
    f"{REPO_HF_PATH}/raw/review_categories/Video_Games.jsonl",
]

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "amazon_video_games"
CSV_OUT    = OUTPUT_DIR / "amazon_video_games_medium.csv"
SUMMARY_OUT= OUTPUT_DIR / "dataset_summary.txt"

# Column normalisation map: raw-name -> standard-name
COL_MAP = {
    "user":        "user_id",
    "user_id":     "user_id",
    "item":        "item_id",
    "item_id":     "item_id",
    "asin":        "item_id",
    "parent_asin": "item_id",
    "rating":      "rating",
    "timestamp":   "timestamp",
}
REQUIRED = ["user_id", "item_id", "rating", "timestamp"]

# Medium-subset parameters
TOP_USERS  = 5_000
TOP_ITEMS  = 3_000
MIN_INTER  = 5


# ---------------------------------------------------------------------------
# Helper: column normalisation
# ---------------------------------------------------------------------------
def normalise_columns(df: pd.DataFrame, source_path: str) -> pd.DataFrame:
    rename = {c: COL_MAP[c] for c in df.columns if c in COL_MAP}
    df = df.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"\nERROR: Required columns missing after normalisation.")
        print(f"       File            : {source_path}")
        print(f"       Available cols  : {list(df.columns)}")
        print(f"       Missing cols    : {missing}")
        sys.exit(1)
    return df[REQUIRED]


# ---------------------------------------------------------------------------
# Helper: find + read dataset
# ---------------------------------------------------------------------------
def find_and_read(fs: HfFileSystem) -> tuple[pd.DataFrame, str]:
    """
    Try each candidate path in order.
    Returns (dataframe, used_path).
    If no candidate works, auto-discovers the closest match.
    """
    # --- Try known candidates ---
    for path in CANDIDATE_PATHS:
        try:
            info = fs.info(path)
            size_mb = info.get("size", 0) / 1e6
            ext = Path(path).suffix.lower()
            print(f"      Found : {path}")
            print(f"      Size  : {size_mb:.1f} MB  |  Format : {ext}")

            if ext == ".csv":
                print("      Reading CSV via HfFileSystem ...")
                with fs.open(path, "rb") as fh:
                    df = pd.read_csv(fh)
            elif ext == ".parquet":
                import pyarrow.parquet as pq
                print("      Reading Parquet via HfFileSystem ...")
                with fs.open(path, "rb") as fh:
                    df = pq.read_table(fh).to_pandas()
            elif ext == ".jsonl":
                print("      Reading JSONL via HfFileSystem (may take a while) ...")
                with fs.open(path, "rb") as fh:
                    df = pd.read_json(fh, lines=True)
            else:
                print(f"      WARNING: Unknown extension '{ext}', skipping.")
                continue

            return df, path

        except FileNotFoundError:
            print(f"      Not found : {path}")
        except Exception as e:
            print(f"      ERROR reading {path}: {e}")

    # --- Auto-discover fallback ---
    print("\n      No candidate matched. Scanning repository for Video_Games files ...")
    all_vg = fs.glob(f"{REPO_HF_PATH}/**/*Video_Games*")
    if not all_vg:
        print("ERROR: No Video_Games files found in the repository.")
        sys.exit(1)

    print(f"      Found {len(all_vg)} file(s):")
    for f in all_vg:
        print(f"        {f}")

    # Pick the first CSV or parquet
    chosen = None
    for f in all_vg:
        if f.endswith((".csv", ".parquet")):
            chosen = f
            break

    if chosen is None:
        print("ERROR: No readable CSV or Parquet file found for Video_Games.")
        sys.exit(1)

    print(f"\n      Auto-selected: {chosen}")
    ext = Path(chosen).suffix.lower()
    with fs.open(chosen, "rb") as fh:
        if ext == ".csv":
            df = pd.read_csv(fh)
        else:
            import pyarrow.parquet as pq
            df = pq.read_table(fh).to_pandas()

    return df, chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # [1/6] Connect
    print("\n" + "="*60)
    print("[1/6] Connecting to Hugging Face")
    print("="*60)
    try:
        fs = HfFileSystem()
        # Quick connectivity check
        fs.ls(REPO_HF_PATH, detail=False)
        print("      Connected successfully.")
    except Exception as e:
        print(f"ERROR: Cannot connect to Hugging Face.\n       {e}")
        print("       Check your internet connection or run: huggingface-cli login")
        sys.exit(1)

    # [2/6] Search
    print("\n" + "="*60)
    print("[2/6] Searching for Video Games dataset files")
    print("="*60)
    print("      Trying candidate paths in priority order ...")

    # [3/6] Read
    print("\n" + "="*60)
    print("[3/6] Reading dataset")
    print("="*60)
    df_raw, used_path = find_and_read(fs)
    original_rows = len(df_raw)
    print(f"\n      Loaded {original_rows:,} rows")
    print(f"      Columns in file : {list(df_raw.columns)}")

    # [4/6] Normalise
    print("\n" + "="*60)
    print("[4/6] Normalising columns")
    print("="*60)
    df = normalise_columns(df_raw, used_path)
    print(f"      Kept columns : {REQUIRED}")
    print(f"      dtypes :")
    for col in REQUIRED:
        print(f"        {col:12s} : {df[col].dtype}")

    # [5/6] Medium subset
    print("\n" + "="*60)
    print("[5/6] Creating medium subset")
    print("="*60)

    # Step A - top active users
    print(f"\n  Step A : selecting top {TOP_USERS:,} most active users ...")
    top_users = df["user_id"].value_counts().nlargest(TOP_USERS).index
    df = df[df["user_id"].isin(top_users)].copy()
    print(f"           -> {len(df):,} rows  |  {df['user_id'].nunique():,} users")

    # Step B - top rated items (within those users)
    print(f"\n  Step B : selecting top {TOP_ITEMS:,} most rated items ...")
    top_items = df["item_id"].value_counts().nlargest(TOP_ITEMS).index
    df = df[df["item_id"].isin(top_items)].copy()
    print(f"           -> {len(df):,} rows  |  {df['item_id'].nunique():,} items")

    # Step C - iterative k-core (k=5)
    print(f"\n  Step C : iterative {MIN_INTER}-core filtering ...")
    iteration = 0
    while True:
        prev = len(df)
        keep_u = df["user_id"].value_counts()
        keep_i = df["item_id"].value_counts()
        df = df[
            df["user_id"].isin(keep_u[keep_u >= MIN_INTER].index) &
            df["item_id"].isin(keep_i[keep_i >= MIN_INTER].index)
        ].copy()
        iteration += 1
        if len(df) == prev:
            break
        print(f"           pass {iteration}: {len(df):,} rows  "
              f"| {df['user_id'].nunique():,} users  "
              f"| {df['item_id'].nunique():,} items")

    n_users    = df["user_id"].nunique()
    n_items    = df["item_id"].nunique()
    final_rows = len(df)
    print(f"\n  Final subset:")
    print(f"    Rows  : {final_rows:,}")
    print(f"    Users : {n_users:,}")
    print(f"    Items : {n_items:,}")

    # [6/6] Save
    print("\n" + "="*60)
    print("[6/6] Saving files")
    print("="*60)

    df.to_csv(CSV_OUT, index=False)
    size_mb = CSV_OUT.stat().st_size / 1e6
    print(f"\n  CSV saved     : {CSV_OUT}")
    print(f"  File size     : {size_mb:.2f} MB")

    # Summary
    config_name = "/".join(used_path.replace(REPO_HF_PATH + "/", "").split("/")[:-1])
    summary = "\n".join([
        f"Dataset name     : Amazon Reviews 2023 - Video Games",
        f"HuggingFace repo : {REPO}",
        f"Config/folder    : {config_name}",
        f"Source file      : {used_path}",
        f"Original rows    : {original_rows:,}",
        f"Final rows       : {final_rows:,}",
        f"Number of users  : {n_users:,}",
        f"Number of items  : {n_items:,}",
        f"Rating min       : {df['rating'].min()}",
        f"Rating max       : {df['rating'].max()}",
        f"Timestamp min    : {int(df['timestamp'].min())}",
        f"Timestamp max    : {int(df['timestamp'].max())}",
        f"Output CSV       : {CSV_OUT}",
    ])
    SUMMARY_OUT.write_text(summary + "\n", encoding="utf-8")
    print(f"  Summary saved : {SUMMARY_OUT}")

    print("\n" + "="*60)
    print("  ALL DONE")
    print("="*60)
    print(summary)
    print("="*60)
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\nTo re-run later:")
    print(f"  python scripts/download_amazon_video_games_medium.py")


if __name__ == "__main__":
    main()
