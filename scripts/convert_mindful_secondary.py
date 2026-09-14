import pandas as pd
import numpy as np
import os

# ============================================================
# Load the touch log
# ============================================================
master_folder = r"C:\doorway_bench\data\raw\mindful\master"
touch_path = os.path.join(master_folder, "Master_TouchLog_All.csv")

print("Loading touch log...")
raw = pd.read_csv(touch_path)
print(f"Loaded {len(raw)} rows")

# ============================================================
# Prepare
# ============================================================
raw['timestamp_sec'] = raw['Timestamp'] / 1000.0
raw = raw.sort_values(['Participant_ID', 'Condition', 'timestamp_sec']).reset_index(drop=True)

# ============================================================
# Build the secondary task DataFrame
# ============================================================
# Each row is one touch event. The label is the condition.
# We keep features that describe the touch dynamics.

secondary = pd.DataFrame({
    'user_id': raw['Participant_ID'],
    'timestamp': raw['timestamp_sec'],
    'event_type': raw['Event_Type'],           # CLICK or SCROLL
    'context': 'tiktok',                        # single app
    'scroll_delta_x': raw['ScrollDeltaX'],
    'scroll_delta_y': raw['ScrollDeltaY'],
    'swipe_distance': raw['Swipe_Distance'],
    'scroll_velocity': raw['Scroll_Velocity'],
    'time_delta_sec': raw['Time_Delta_Sec'],
    'forgetting_proxy': (raw['Condition'] == 'Mindless').astype(int),
    'label_source': 'mindful_mindless',
    'confidence': 1.0,
    'dataset': 'mindful',
})

# ============================================================
# Save
# ============================================================
os.makedirs(r"C:\doorway_bench\data\processed", exist_ok=True)
output_path = r"C:\doorway_bench\data\processed\mindful_secondary.parquet"
secondary.to_parquet(output_path, index=False)

print(f"\nSaved {len(secondary)} touch events to {output_path}")
print(f"\nFirst 10 rows:")
print(secondary.head(10))
print(f"\nRows per label:")
print(secondary.groupby('forgetting_proxy').size())
print(f"\nUnique users: {secondary['user_id'].nunique()}")
print(f"\nEvent type counts:")
print(secondary['event_type'].value_counts())
print(f"\nMissing values:")
print(secondary.isnull().sum())