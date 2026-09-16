import pandas as pd
import os

# CHANGE THIS to the actual participant folder name
participant = "p1"

root = r"C:\doorway_bench\data\raw\rlkwic"
folder = os.path.join(root, participant)

print(f"Inspecting: {folder}\n")

for filename in sorted(os.listdir(folder)):
    if filename.endswith(".csv"):
        filepath = os.path.join(folder, filename)
        print("=" * 70)
        print(f"FILE: {filename}")
        try:
            df = pd.read_csv(filepath, nrows=5)
            print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
            print("First 5 rows:")
            print(df.head(5).to_string())
            # Count rows
            full = pd.read_csv(filepath, usecols=[0])
            print(f"Total rows: {len(full)}")
        except Exception as e:
            print(f"Error: {e}")
        print()