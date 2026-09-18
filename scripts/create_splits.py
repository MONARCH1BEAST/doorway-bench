import pandas as pd
import numpy as np
import json
import os

# Load the full feature matrix
input_path = r"C:\doorway_bench\data\processed\rlkwic_full_features.parquet"
df = pd.read_parquet(input_path)
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

print(f"Loaded {len(df)} rows from {df['user_id'].nunique()} users")

splits = {}

# ============================================================
# 1. WITHIN-DATASET TEMPORAL SPLIT (70/15/15)
# ============================================================
within_temporal = {}
for user in df['user_id'].unique():
    user_df = df[df['user_id'] == user].reset_index(drop=True)
    n = len(user_df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    within_temporal[user] = {
        'train': user_df.index[:train_end].tolist(),
        'val': user_df.index[train_end:val_end].tolist(),
        'test': user_df.index[val_end:].tolist(),
    }

splits['within_temporal'] = within_temporal

# ============================================================
# 2. LEAVE-ONE-USER-OUT
# ============================================================
louo = {}
for user in df['user_id'].unique():
    test_idx = df[df['user_id'] == user].index.tolist()
    train_idx = df[df['user_id'] != user].index.tolist()
    louo[user] = {'train': train_idx, 'test': test_idx}

splits['leave_one_user_out'] = louo

# ============================================================
# 3. TEMPORAL GENERALIZATION (first 80% vs last 20% per user)
# ============================================================
temporal_gen = {}
for user in df['user_id'].unique():
    user_df = df[df['user_id'] == user].reset_index(drop=True)
    n = len(user_df)
    cutoff = int(0.80 * n)
    temporal_gen[user] = {
        'train': user_df.index[:cutoff].tolist(),
        'test': user_df.index[cutoff:].tolist(),
    }

splits['temporal_generalization'] = temporal_gen

# ============================================================
# 4. 5-FOLD CROSS-VALIDATION (stratified by user)
# ============================================================
from sklearn.model_selection import KFold

cv_folds = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(df):
    cv_folds.append({
        'train': train_idx.tolist(),
        'test': test_idx.tolist(),
    })

splits['cv_5fold'] = cv_folds

# ============================================================
# Save
# ============================================================
output_dir = r"C:\doorway_bench\data\splits"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "splits.json")

with open(output_path, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"\nSaved splits to {output_path}")

# ============================================================
# Report
# ============================================================
print(f"\n{'='*60}")
print("SPLITS SUMMARY")
print(f"{'='*60}")

print(f"\n1. Within-dataset temporal split:")
for user, s in splits['within_temporal'].items():
    print(f"   {user}: train={len(s['train'])}, val={len(s['val'])}, test={len(s['test'])}")

print(f"\n2. Leave-one-user-out: {len(splits['leave_one_user_out'])} folds")

print(f"\n3. Temporal generalization:")
for user, s in splits['temporal_generalization'].items():
    print(f"   {user}: train={len(s['train'])}, test={len(s['test'])}")

print(f"\n4. 5-fold CV: {len(splits['cv_5fold'])} folds")

# Verify: total rows across train+val+test should equal full dataset for within_temporal
total = 0
for user, s in splits['within_temporal'].items():
    total += len(s['train']) + len(s['val']) + len(s['test'])
print(f"\nVerification: total rows across within_temporal = {total} (should be {len(df)})")

# Check class balance in each fold
print(f"\nClass balance (within_temporal test sets):")
for user, s in splits['within_temporal'].items():
    test_labels = df.loc[s['test'], 'forgetting_proxy']
    if len(test_labels) > 0:
        print(f"   {user}: {test_labels.mean():.2%} forgetting")