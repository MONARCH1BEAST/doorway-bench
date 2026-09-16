import pandas as pd
import numpy as np
import os

# Load current RLKWiC
df = pd.read_parquet(r"C:\doorway_bench\data\processed\rlkwic.parquet")
print(f"Loaded {len(df)} rows")

# ============================================================
# FIX 1: Remove rows where source == target (not real switches)
# ============================================================
before = len(df)
df = df[df['source_context'] != df['target_context']].reset_index(drop=True)
after = len(df)
print(f"\nFix 1: Removed {before - after} non-switches (source == target)")
print(f"Remaining: {after} rows")

# ============================================================
# FIX 2: Map context IDs to human-readable labels
# ============================================================
# We need to load contexts.csv from each participant
root = r"C:\doorway_bench\data\raw\rlkwic"
participants = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

# Build a global lookup: (participant, context_id) -> label
context_lookup = {}
for p in participants:
    ctx_path = os.path.join(root, p, "contexts.csv")
    if os.path.exists(ctx_path):
        ctx = pd.read_csv(ctx_path)
        for _, row in ctx.iterrows():
            context_lookup[(p, str(row['id']))] = str(row['label'])

print(f"\nFix 2: Built context lookup with {len(context_lookup)} entries")

# Map source and target
df['source_context'] = df.apply(
    lambda r: context_lookup.get((r['user_id'], str(r['source_context'])), 
                                  f"ctx_{r['source_context']}"),
    axis=1
)
df['target_context'] = df.apply(
    lambda r: context_lookup.get((r['user_id'], str(r['target_context'])), 
                                  f"ctx_{r['target_context']}"),
    axis=1
)

# ============================================================
# Re-sort and recompute dwell_time_after (since we removed rows)
# ============================================================
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
df['next_timestamp'] = df.groupby('user_id')['timestamp'].shift(-1)
df['dwell_time_after'] = df['next_timestamp'] - df['timestamp']
df = df.drop(columns=['next_timestamp'])

# ============================================================
# Save
# ============================================================
output_path = r"C:\doorway_bench\data\processed\rlkwic.parquet"
try:
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
except:
    output_path = r"C:\doorway_bench\data\processed\rlkwic.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

# ============================================================
# Report
# ============================================================
print(f"\n{'='*60}")
print("FINAL RLKWiC SUMMARY")
print(f"{'='*60}")
print(f"Total rows: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"\nForgetting distribution:")
print(df['forgetting_proxy'].value_counts())
print(f"\nFirst 10 rows:")
print(df[['user_id', 'timestamp', 'source_context', 'target_context',
          'dwell_time_before', 'dwell_time_after', 'forgetting_proxy']].head(10).to_string())
print(f"\nTop 10 source contexts:")
print(df['source_context'].value_counts().head(10))
print(f"\nMissing values:")
print(df.isnull().sum())