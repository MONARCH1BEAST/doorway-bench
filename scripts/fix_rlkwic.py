import pandas as pd
import numpy as np
import os

# Load the current RLKWiC data
input_path = r"C:\doorway_bench\data\processed\rlkwic.parquet"
df = pd.read_parquet(input_path)

print(f"Loaded {len(df)} rows")

# Step 1: Sort by user and timestamp
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# Step 2: Compute dwell_time_after for each row
# For each row, it's the time until the next row for the same user
df['next_timestamp'] = df.groupby('user_id')['timestamp'].shift(-1)
df['dwell_time_after'] = df['next_timestamp'] - df['timestamp']

# Drop the helper column
df = df.drop(columns=['next_timestamp'])

# Step 3: Save
output_path = r"C:\doorway_bench\data\processed\rlkwic.parquet"
try:
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
except:
    output_path = r"C:\doorway_bench\data\processed\rlkwic.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

# Verify
print(f"\nAfter sorting:")
print(f"Total rows: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"\nForgetting distribution:")
print(df['forgetting_proxy'].value_counts())
print(f"\nFirst 10 rows (should be sorted by timestamp):")
print(df[['user_id', 'timestamp', 'source_context', 'target_context', 
          'dwell_time_before', 'dwell_time_after', 'forgetting_proxy']].head(10))
print(f"\nMissing values:")
print(df.isnull().sum())

# Check: per user, is timestamp sorted?
print(f"\nSorting check per user:")
for user in df['user_id'].unique():
    sub = df[df['user_id'] == user]
    is_sorted = sub['timestamp'].is_monotonic_increasing
    print(f"  {user}: {len(sub)} rows, sorted = {is_sorted}")