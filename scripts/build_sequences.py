import pandas as pd
import numpy as np
import os
import json
import pickle

# Load full feature matrix
processed = r"C:\doorway_bench\data\processed"
df = pd.read_parquet(os.path.join(processed, "rlkwic_modeling_ready.parquet"))
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# Fill NaNs
for col in df.columns:
    if df[col].isnull().any() and df[col].dtype != 'object':
        if col not in ['forgetting_proxy']:
            df[col] = df[col].fillna(df[col].median())

# Feature columns (same as before, minus non-features)
non_features = ['user_id', 'timestamp', 'event_type', 'source_context',
                'target_context', 'device', 'label_source', 'confidence',
                'dataset', 'forgetting_proxy']
feature_cols = [c for c in df.columns if c not in non_features]

print(f"Features: {len(feature_cols)}")
print(f"Rows: {len(df)}")

# ============================================================
# Build sequences per user
# ============================================================
WINDOW = 5  # look at previous 5 switches
sequences = []  # list of (user_id, row_idx, X_seq, y)
labels = []
indices = []

for user, group in df.groupby('user_id'):
    group = group.reset_index(drop=True)
    
    for i in range(len(group)):
        # Need at least WINDOW previous events
        start = max(0, i - WINDOW)
        seq = group.iloc[start:i+1][feature_cols].values  # (≤WINDOW+1, n_features)
        
        # Pad if fewer than WINDOW+1 rows
        if seq.shape[0] < WINDOW + 1:
            pad = np.zeros((WINDOW + 1 - seq.shape[0], len(feature_cols)))
            seq = np.vstack([pad, seq])
        
        sequences.append(seq)
        labels.append(group.iloc[i]['forgetting_proxy'])
        indices.append(group.iloc[i].name if hasattr(group.iloc[i], 'name') else i)

# Get original indices from df
original_indices = []
counter = 0
for user, group in df.groupby('user_id'):
    group_orig = group.index.tolist()
    for i in range(len(group_orig)):
        original_indices.append(group_orig[i])

X_seq = np.array(sequences)
y_seq = np.array(labels)
idx_seq = np.array(original_indices)

print(f"\nSequence tensor shape: {X_seq.shape}")
print(f"Labels shape: {y_seq.shape}")
print(f"Positive rate: {y_seq.mean():.2%}")

# ============================================================
# Save
# ============================================================
seq_dir = r"C:\doorway_bench\data\processed\sequences"
os.makedirs(seq_dir, exist_ok=True)

np.save(os.path.join(seq_dir, "X_seq.npy"), X_seq)
np.save(os.path.join(seq_dir, "y_seq.npy"), y_seq)
np.save(os.path.join(seq_dir, "idx_seq.npy"), idx_seq)

with open(os.path.join(seq_dir, "feature_names.pkl"), 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"\nSaved sequences to {seq_dir}")
print(f"  X_seq.npy: {X_seq.shape}")
print(f"  y_seq.npy: {y_seq.shape}")
print(f"  idx_seq.npy: {idx_seq.shape}")

# Show one example
print(f"\nExample sequence (first row):")
print(X_seq[0])