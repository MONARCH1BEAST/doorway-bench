import pandas as pd
import numpy as np
import os
from scipy.stats import entropy

# Load data with temporal features
input_path = r"C:\doorway_bench\data\processed\rlkwic_with_temporal.parquet"
df = pd.read_parquet(input_path)
print(f"Loaded {len(df)} rows")

df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ============================================================
# 1. context_dissimilarity (via co-occurrence)
# ============================================================
# Two contexts are "similar" if they tend to appear close in time.
# Build a co-occurrence matrix per user within a 30-min window.

# First, collect all unique contexts
all_contexts = sorted(set(df['source_context']) | set(df['target_context']))
context_to_idx = {c: i for i, c in enumerate(all_contexts)}
n_contexts = len(all_contexts)

# Build co-occurrence counts
cooc = np.zeros((n_contexts, n_contexts))

for user, group in df.groupby('user_id'):
    group = group.sort_values('timestamp').reset_index(drop=True)
    ts = group['timestamp'].values
    ctx = group['target_context'].values
    for i in range(len(group)):
        for j in range(i+1, min(i+20, len(group))):
            if ts[j] - ts[i] > 1800:  # 30 minutes
                break
            ci = context_to_idx[ctx[i]]
            cj = context_to_idx[ctx[j]]
            cooc[ci, cj] += 1
            cooc[cj, ci] += 1

# Normalize to get similarity in [0, 1]
row_sums = cooc.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
similarity = cooc / row_sums

# Compute dissimilarity = 1 - similarity
dissimilarities = []
for _, row in df.iterrows():
    src = row['source_context']
    tgt = row['target_context']
    if src in context_to_idx and tgt in context_to_idx:
        i, j = context_to_idx[src], context_to_idx[tgt]
        sim = similarity[i, j]
        dissimilarities.append(1.0 - sim)
    else:
        dissimilarities.append(1.0)

df['context_dissimilarity'] = dissimilarities

# ============================================================
# 2. context_entropy_5min (same as before)
# ============================================================
context_entropies = []
for user, group in df.groupby('user_id'):
    group = group.reset_index(drop=True)
    entropies = []
    for i in range(len(group)):
        t = group['timestamp'].iloc[i]
        window = group[(group['timestamp'] >= t - 300) & (group['timestamp'] <= t)]
        if len(window) < 2:
            entropies.append(0.0)
        else:
            counts = window['target_context'].value_counts(normalize=True)
            entropies.append(entropy(counts))
    context_entropies.extend(entropies)

df['context_entropy_5min'] = context_entropies

# ============================================================
# 3. novelty (same as before)
# ============================================================
novelties = []
for user, group in df.groupby('user_id'):
    group = group.reset_index(drop=True)
    seen = set()
    nov = []
    for _, row in group.iterrows():
        nov.append(0 if row['target_context'] in seen else 1)
        seen.add(row['target_context'])
    novelties.extend(nov)

df['novelty'] = novelties

# ============================================================
# 4. return_probability (FIXED)
# ============================================================
# Define: P(return to source | source was just left)
# = number of times source_context appears as target / total switches by user
# This is bounded in [0, 1].

return_probs = {}
for user, group in df.groupby('user_id'):
    total = len(group)
    target_counts = group['target_context'].value_counts()
    for ctx in group['source_context'].unique():
        return_probs[(user, ctx)] = target_counts.get(ctx, 0) / total

df['return_probability'] = [
    return_probs.get((row['user_id'], row['source_context']), 0.0)
    for _, row in df.iterrows()
]

# ============================================================
# Save
# ============================================================
output_path = r"C:\doorway_bench\data\processed\rlkwic_with_features.parquet"
try:
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
except:
    output_path = r"C:\doorway_bench\data\processed\rlkwic_with_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

# ============================================================
# Report
# ============================================================
print(f"\n{'='*60}")
print("CONTEXTUAL FEATURES (v2)")
print(f"{'='*60}")
print(f"Total rows: {len(df)}")
print(f"\nSummary statistics:")
print(df[['context_dissimilarity', 'context_entropy_5min',
          'novelty', 'return_probability']].describe().to_string())
print(f"\nMissing values:")
print(df[['context_dissimilarity', 'context_entropy_5min',
          'novelty', 'return_probability']].isnull().sum())
print(f"\nFirst 10 rows:")
print(df[['user_id', 'source_context', 'target_context',
          'context_dissimilarity', 'context_entropy_5min',
          'novelty', 'return_probability', 'forgetting_proxy']].head(10).to_string())