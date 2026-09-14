import pandas as pd
import numpy as np
import os

# ============================================================
# STEP 1: Load the raw touch log
# ============================================================
master_folder = r"C:\doorway_bench\data\raw\mindful\master"
touch_path = os.path.join(master_folder, "Master_TouchLog_All.csv")

print("Loading touch log...")
raw = pd.read_csv(touch_path)
print(f"Loaded {len(raw)} rows")

# ============================================================
# STEP 2: Clean and prepare
# ============================================================

# Convert timestamp from milliseconds to seconds
raw['timestamp_sec'] = raw['Timestamp'] / 1000.0

# Make sure it's sorted by participant and time
raw = raw.sort_values(['Participant_ID', 'timestamp_sec']).reset_index(drop=True)

# Drop rows with missing PackageName
raw = raw.dropna(subset=['PackageName'])

# ============================================================
# STEP 3: Detect context switches
# ============================================================
# A context switch happens when the PackageName changes
# from one event to the next, for the same participant.

switches = []

for participant_id, group in raw.groupby('Participant_ID'):
    group = group.sort_values('timestamp_sec').reset_index(drop=True)
    
    # Find rows where PackageName changes
    group['prev_package'] = group['PackageName'].shift(1)
    group['prev_time'] = group['timestamp_sec'].shift(1)
    
    # A switch is where prev_package exists and differs from current
    switch_mask = (group['prev_package'].notna()) & (group['prev_package'] != group['PackageName'])
    switch_rows = group[switch_mask].copy()
    
    if len(switch_rows) == 0:
        continue
    
    # For each switch, compute dwell time in the previous context
    # We need to find when the previous context was entered
    switch_rows['dwell_time_before'] = switch_rows['timestamp_sec'] - switch_rows['prev_time']
    
    # Build a DataFrame for this participant's switches
    participant_switches = pd.DataFrame({
        'user_id': participant_id,
        'timestamp': switch_rows['timestamp_sec'].values,
        'event_type': 'app_switch',
        'source_context': switch_rows['prev_package'].values,
        'target_context': switch_rows['PackageName'].values,
        'dwell_time_before': switch_rows['dwell_time_before'].values,
        'dwell_time_after': np.nan,  # We don't know yet
        'device': 'mobile',
        'forgetting_proxy': (switch_rows['Condition'] == 'Mindless').astype(int).values,
        'label_source': 'mindful_mindless',
        'confidence': 1.0,
        'dataset': 'mindful',
    })
    
    switches.append(participant_switches)

# Combine all participants
if len(switches) == 0:
    print("ERROR: No switches detected. Check the data.")
    exit(1)

unified = pd.concat(switches, ignore_index=True)

# ============================================================
# STEP 4: Save
# ============================================================
os.makedirs(r"C:\doorway_bench\data\processed", exist_ok=True)
output_path = r"C:\doorway_bench\data\processed\mindful.parquet"
unified.to_parquet(output_path, index=False)

print(f"\nSaved {len(unified)} switches to {output_path}")
print(f"\nColumns: {unified.columns.tolist()}")
print(f"\nFirst 10 rows:")
print(unified.head(10))
print(f"\nRows per condition (forgetting_proxy):")
print(unified.groupby('forgetting_proxy').size())
print(f"\nUnique users: {unified['user_id'].nunique()}")
print(f"\nTop source contexts:")
print(unified['source_context'].value_counts().head(10))
print(f"\nTop target contexts:")
print(unified['target_context'].value_counts().head(10))