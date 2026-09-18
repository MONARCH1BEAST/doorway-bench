import pandas as pd
import numpy as np
import os

# Load RLKWiC
input_path = r"C:\doorway_bench\data\processed\rlkwic.parquet"
df = pd.read_parquet(input_path)

print(f"Loaded {len(df)} rows")
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ============================================================
# 1. time_since_last_switch
# ============================================================
df['time_since_last_switch'] = df.groupby('user_id')['timestamp'].diff()

# ============================================================
# 2. switch_rate_5min and switch_rate_30min
# ============================================================
def rolling_count(timestamps, window_seconds):
    """Count how many previous events fall within window_seconds of each event."""
    counts = np.zeros(len(timestamps))
    for i in range(len(timestamps)):
        t = timestamps.iloc[i]
        # Count events in [t - window, t]
        mask = (timestamps >= t - window_seconds) & (timestamps <= t)
        counts[i] = mask.sum() - 1  # subtract self
    return counts

switch_rate_5min = []
switch_rate_30min = []

for user, group in df.groupby('user_id'):
    ts = group['timestamp'].reset_index(drop=True)
    switch_rate_5min.extend(rolling_count(ts, 300).tolist())
    switch_rate_30min.extend(rolling_count(ts, 1800).tolist())

df['switch_rate_5min'] = switch_rate_5min
df['switch_rate_30min'] = switch_rate_30min

# ============================================================
# 3. switch_cascade_length
# ============================================================
# Cascade: consecutive switches within 60 seconds of each other
cascade_lengths = []
for user, group in df.groupby('user_id'):
    group = group.reset_index(drop=True)
    lengths = []
    current = 1
    for i in range(len(group)):
        if i == 0:
            current = 1
        else:
            gap = group['timestamp'].iloc[i] - group['timestamp'].iloc[i-1]
            if gap < 60:
                current += 1
            else:
                current = 1
        lengths.append(current)
    cascade_lengths.extend(lengths)

df['switch_cascade_length'] = cascade_lengths

# ============================================================
# 4. Time of day features
# ============================================================
# Unix seconds → hour of day (UTC)
seconds_in_day = 86400
df['hour_of_day'] = (df['timestamp'] % seconds_in_day) / 3600.0
df['time_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['time_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

# ============================================================
# Save
# ============================================================
output_path = r"C:\doorway_bench\data\processed\rlkwic_with_temporal.parquet"
try:
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
except:
    output_path = r"C:\doorway_bench\data\processed\rlkwic_with_temporal.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

# Report
print(f"\n{'='*60}")
print("TEMPORAL FEATURES SUMMARY")
print(f"{'='*60}")
print(f"Total rows: {len(df)}")
print(f"\nNew columns: time_since_last_switch, switch_rate_5min, "
      f"switch_rate_30min, switch_cascade_length, hour_of_day, "
      f"time_of_day_sin, time_of_day_cos")
print(f"\nSummary statistics:")
print(df[['time_since_last_switch', 'switch_rate_5min', 'switch_rate_30min',
          'switch_cascade_length', 'hour_of_day']].describe().to_string())
print(f"\nMissing values in new columns:")
print(df[['time_since_last_switch', 'switch_rate_5min', 'switch_rate_30min',
          'switch_cascade_length', 'hour_of_day']].isnull().sum())
print(f"\nFirst 5 rows:")
print(df[['user_id', 'timestamp', 'time_since_last_switch', 'switch_rate_5min',
          'switch_rate_30min', 'switch_cascade_length', 'hour_of_day',
          'forgetting_proxy']].head().to_string())