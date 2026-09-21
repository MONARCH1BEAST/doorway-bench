import pandas as pd
import os

processed = r"C:\doorway_bench\data\processed"

print("Combining all processed datasets...")
print("=" * 60)

# Load each processed dataset
files = {
    'rlkwic': 'rlkwic_full_features.parquet',
    'mindful': 'mindful_secondary.parquet',
    'kaggle': 'kaggle_reference.parquet',
}

datasets = {}
for name, filename in files.items():
    path = os.path.join(processed, filename)
    if os.path.exists(path):
        datasets[name] = pd.read_parquet(path)
        print(f"Loaded {name}: {len(datasets[name])} rows")
    else:
        print(f"MISSING: {filename}")

# Save individual files summary
print(f"\n{'='*60}")
print("DATASET SUMMARY")
print(f"{'='*60}")
for name, df in datasets.items():
    print(f"\n{name}:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    if 'forgetting_proxy' in df.columns:
        print(f"  Forgetting rate: {df['forgetting_proxy'].mean():.2%}")
    if 'user_id' in df.columns:
        print(f"  Users: {df['user_id'].nunique()}")

# Save a manifest
manifest = pd.DataFrame([
    {'dataset': name, 'rows': len(df), 'columns': len(df.columns)}
    for name, df in datasets.items()
])
manifest.to_csv(os.path.join(processed, "combined_manifest.csv"), index=False)

print(f"\n{'='*60}")
print(f"Manifest saved to {processed}/combined_manifest.csv")
print(f"Total datasets: {len(datasets)}")
print(f"Total rows: {sum(len(df) for df in datasets.values())}")