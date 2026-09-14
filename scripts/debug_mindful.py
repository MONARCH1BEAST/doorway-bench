import pandas as pd
import os

master_folder = r"C:\doorway_bench\data\raw\mindful\master"
touch_path = os.path.join(master_folder, "Master_TouchLog_All.csv")

print("Loading...")
raw = pd.read_csv(touch_path)
raw['timestamp_sec'] = raw['Timestamp'] / 1000.0
raw = raw.sort_values(['Participant_ID', 'Condition', 'timestamp_sec']).reset_index(drop=True)

# How many unique packages per participant per condition?
print("\n=== Unique packages per participant/condition ===")
summary = raw.groupby(['Participant_ID', 'Condition'])['PackageName'].nunique()
print(summary)

# What are the actual package names?
print("\n=== Value counts of PackageName ===")
print(raw['PackageName'].value_counts())

# For one participant, print first 30 rows to see if package ever changes
print("\n=== First 30 rows for P-001, Mindful ===")
sample = raw[(raw['Participant_ID'] == 'P-001') & (raw['Condition'] == 'Mindful')].head(30)
print(sample[['timestamp_sec', 'Event_Type', 'PackageName']].to_string())

# Check: for each participant/condition, how many times does PackageName change?
print("\n=== Number of package changes per participant/condition ===")
for (pid, cond), group in raw.groupby(['Participant_ID', 'Condition']):
    changes = (group['PackageName'] != group['PackageName'].shift(1)).sum() - 1
    print(f"{pid} / {cond}: {changes} changes")