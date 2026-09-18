import pandas as pd
import numpy as np
import os
import networkx as nx
from scipy.stats import entropy

# Load data with contextual features
input_path = r"C:\doorway_bench\data\processed\rlkwic_with_features.parquet"
df = pd.read_parquet(input_path)
print(f"Loaded {len(df)} rows")

df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

# 1. switch_velocity: switches per minute within a 30-min window
switch_velocity = []
for user, group in df.groupby('user_id'):
    group = group.reset_index(drop=True)
    vels = []
    for i in range(len(group)):
        t = group['timestamp'].iloc[i]
        window = group[(group['timestamp'] >= t - 1800) & (group['timestamp'] <= t)]
        elapsed_min = max((t - window['timestamp'].min()) / 60.0, 1.0)
        vels.append(len(window) / elapsed_min)
    switch_velocity.extend(vels)

df['switch_velocity'] = switch_velocity

# 2. revisit_count: how many times source_context has appeared as target before
revisit_counts = []
for user, group in df.groupby('user_id'):
    group = group.reset_index(drop=True)
    seen_counts = {}
    revs = []
    for _, row in group.iterrows():
        ctx = row['source_context']
        revs.append(seen_counts.get(ctx, 0))
        seen_counts[ctx] = seen_counts.get(ctx, 0) + 1
    revisit_counts.extend(revs)

df['revisit_count'] = revisit_counts

# 3. session_position: normalized position in user's timeline
session_positions = []
for user, group in df.groupby('user_id'):
    n = len(group)
    if n == 1:
        session_positions.append(0.5)
    else:
        session_positions.extend([i / (n - 1) for i in range(n)])

df['session_position'] = session_positions

# ============================================================
# GRAPH FEATURES (from RLKWiC's own transition network)
# ============================================================

# Build a directed weighted graph per user
source_degree = []
target_degree = []
source_clustering = []
target_clustering = []
edge_weight = []
network_density = []
network_entropy = []

for user, group in df.groupby('user_id'):
    G = nx.DiGraph()
    for _, row in group.iterrows():
        src, tgt = row['source_context'], row['target_context']
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += 1
        else:
            G.add_edge(src, tgt, weight=1)
    
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected)
    density = nx.density(G)
    
    # Entropy of edge weights
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    if len(weights) > 0 and sum(weights) > 0:
        probs = np.array(weights) / sum(weights)
        net_entropy = entropy(probs)
    else:
        net_entropy = 0.0
    
    for _, row in group.iterrows():
        src, tgt = row['source_context'], row['target_context']
        source_degree.append(G.out_degree(src) if src in G else 0)
        target_degree.append(G.in_degree(tgt) if tgt in G else 0)
        source_clustering.append(clustering.get(src, 0.0))
        target_clustering.append(clustering.get(tgt, 0.0))
        edge_weight.append(G[src][tgt]['weight'] if G.has_edge(src, tgt) else 0)
        network_density.append(density)
        network_entropy.append(net_entropy)

df['source_degree'] = source_degree
df['target_degree'] = target_degree
df['source_clustering'] = source_clustering
df['target_clustering'] = target_clustering
df['edge_weight'] = edge_weight
df['network_density'] = network_density
df['network_entropy'] = network_entropy

# ============================================================
# Save
# ============================================================
output_path = r"C:\doorway_bench\data\processed\rlkwic_full_features.parquet"
try:
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
except:
    output_path = r"C:\doorway_bench\data\processed\rlkwic_full_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

# ============================================================
# Report
# ============================================================
print(f"\n{'='*60}")
print("BEHAVIORAL + GRAPH FEATURES")
print(f"{'='*60}")
print(f"Total rows: {len(df)}")
new_cols = ['switch_velocity', 'revisit_count', 'session_position',
            'source_degree', 'target_degree', 'source_clustering',
            'target_clustering', 'edge_weight', 'network_density', 'network_entropy']
print(f"\nNew columns: {new_cols}")
print(f"\nSummary statistics:")
print(df[new_cols].describe().to_string())
print(f"\nMissing values:")
print(df[new_cols].isnull().sum())
print(f"\nAll columns now: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df[['user_id', 'source_context', 'target_context',
          'switch_velocity', 'revisit_count', 'source_degree', 'target_degree',
          'edge_weight', 'network_density', 'network_entropy',
          'forgetting_proxy']].head().to_string())