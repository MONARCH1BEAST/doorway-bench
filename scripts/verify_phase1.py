import pandas as pd
import os

processed = r"C:\doorway_bench\data\processed"

print("=" * 60)
print("PHASE 1 VERIFICATION")
print("=" * 60)

# Check each file exists
files = {
    "RLKWiC": "rlkwic.parquet",
    "Mindful (secondary)": "mindful_secondary.parquet",
    "Kaggle Reference": "kaggle_reference.parquet",
    "App Networks Features": "app_networks_features.csv",
}

for name, filename in files.items():
    path = os.path.join(processed, filename)
    if os.path.exists(path):
        if filename.endswith('.parquet'):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        print(f"\n{name}:")
        print(f"  File: {filename}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Columns: {df.columns.tolist()}")
    else:
        print(f"\n{name}: MISSING ({filename})")

# Check splits
splits_path = r"C:\doorway_bench\data\splits\splits.json"
if os.path.exists(splits_path):
    print(f"\nSplits file exists: {splits_path}")
else:
    print(f"\nSplits file: NOT YET CREATED (we do this in Phase 2)")

# Check raw data folders
raw = r"C:\doorway_bench\data\raw"
print(f"\nRaw data folders:")
for folder in os.listdir(raw):
    folder_path = os.path.join(raw, folder)
    if os.path.isdir(folder_path):
        print(f"  {folder}/")