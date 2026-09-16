import pandas as pd
import numpy as np
import os
import ast

# Root folder containing participant subfolders
root = r"C:\doorway_bench\data\raw\rlkwic"

# Output folder
output_dir = r"C:\doorway_bench\data\processed"
os.makedirs(output_dir, exist_ok=True)

# Find all participant folders (subdirectories)
participants = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
print(f"Found {len(participants)} participants: {participants}")

all_switches = []

for participant in participants:
    folder = os.path.join(root, participant)
    events_path = os.path.join(folder, "events.csv")
    sessions_path = os.path.join(folder, "sessions.csv")
    
    if not os.path.exists(events_path) or not os.path.exists(sessions_path):
        print(f"Skipping {participant} — missing files")
        continue
    
    # Load events and sort by timestamp
    events = pd.read_csv(events_path)
    events = events.sort_values('timestamp_received').reset_index(drop=True)
    
    # Load sessions
    sessions = pd.read_csv(sessions_path)
    
    # For each session, get the first event ID from events_list
    for _, session in sessions.iterrows():
        events_list_str = session['events_list']
        if pd.isna(events_list_str) or events_list_str == '[]':
            continue
        
        try:
            event_ids = ast.literal_eval(events_list_str)
        except:
            continue
        
        if len(event_ids) == 0:
            continue
        
        first_event_id = event_ids[0]
        
        # Find that event in events
        event_row = events[events['id'] == first_event_id]
        if len(event_row) == 0:
            continue
        
        event_row = event_row.iloc[0]
        idx = events[events['id'] == first_event_id].index[0]
        
        # Get previous event for source context
        if idx > 0:
            prev_event = events.iloc[idx - 1]
            source_context = prev_event['selected_context_id']
            prev_time = prev_event['timestamp_received']
        else:
            source_context = np.nan
            prev_time = np.nan
        
        target_context = event_row['selected_context_id']
        timestamp = event_row['timestamp_received'] / 1000.0  # convert to seconds
        
        # Dwell time before: time between previous event and this event
        if not np.isnan(prev_time):
            dwell_time_before = (event_row['timestamp_received'] - prev_time) / 1000.0
        else:
            dwell_time_before = np.nan
        
        # Label: in_context == False means forgetting
        # in_context might be True/False or 1/0
        in_ctx = session['in_context']
        if isinstance(in_ctx, str):
            in_ctx = in_ctx.lower() == 'true'
        forgetting = 0 if in_ctx else 1
        
        all_switches.append({
            'user_id': participant,
            'timestamp': timestamp,
            'event_type': 'context_switch',
            'source_context': source_context,
            'target_context': target_context,
            'dwell_time_before': dwell_time_before,
            'dwell_time_after': np.nan,  # we'll leave this for now
            'device': 'desktop',
            'forgetting_proxy': forgetting,
            'label_source': 'rlkwic_in_context',
            'confidence': 0.8,
            'dataset': 'rlkwic',
        })

if len(all_switches) == 0:
    print("No switches found. Check the data.")
    exit(1)

unified = pd.DataFrame(all_switches)

# Save
output_path = os.path.join(output_dir, "rlkwic.parquet")
try:
    unified.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
except:
    output_path = os.path.join(output_dir, "rlkwic.csv")
    unified.to_csv(output_path, index=False)
    print(f"Parquet failed, saved to {output_path}")

print(f"\nTotal switches: {len(unified)}")
print(f"Unique users: {unified['user_id'].nunique()}")
print(f"\nForgetting distribution:")
print(unified['forgetting_proxy'].value_counts())
print(f"\nFirst 10 rows:")
print(unified.head(10))
print(f"\nMissing values:")
print(unified.isnull().sum())