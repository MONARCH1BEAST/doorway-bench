import pandas as pd
import os

folder = r"C:\doorway_bench\data\raw\kaggle"

# List all files
print("Files in folder:")
for f in os.listdir(folder):
    print(f"  {f}")

# Find the CSV (try both with and without .csv extension)
csv_path = None
for f in os.listdir(folder):
    if f.endswith(".csv"):
        csv_path = os.path.join(folder, f)
        break

if csv_path is None:
    print("No CSV found!")
    exit(1)

print(f"\nLoading: {csv_path}")

df = pd.read_csv(csv_path)

print(f"\nShape: {df.shape}")
print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nUnique users/dates:")
for col in df.columns:
    if 'user' in col.lower() or 'id' in col.lower():
        print(f"  {col}: {df[col].nunique()} unique")