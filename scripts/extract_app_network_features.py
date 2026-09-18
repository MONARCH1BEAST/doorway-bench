import os
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import entropy

folder = r"C:\doorway_bench\data\raw\networkdata\graphs"
output_path = r"C:\doorway_bench\data\processed\app_networks_features.csv"

gml_files = sorted([f for f in os.listdir(folder) if f.endswith('.gml')])
print(f"Processing {len(gml_files)} graphs...")

all_features = []

for fname in gml_files:
    user_id = fname.replace('.gml', '')
    filepath = os.path.join(folder, fname)
    G = nx.read_gml(filepath)
    
    # Basic stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Degrees (weighted and unweighted)
    degrees = [d for _, d in G.degree()]
    weighted_degrees = [d for _, d in G.degree(weight='weight')]
    
    # Clustering (on undirected version)
    G_undirected = G.to_undirected()
    avg_clustering = nx.average_clustering(G_undirected)
    transitivity = nx.transitivity(G_undirected)
    
    # Assortativity
    try:
        degree_assortativity = nx.degree_assortativity_coefficient(G)
    except:
        degree_assortativity = np.nan
    
    # Entropy of edge weights
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    if len(weights) > 0 and sum(weights) > 0:
        weight_probs = np.array(weights) / sum(weights)
        weight_entropy = entropy(weight_probs)
    else:
        weight_entropy = 0.0
    
    # Entropy of degree distribution
    if len(degrees) > 0:
        degree_counts = np.bincount(degrees)
        degree_probs = degree_counts / degree_counts.sum()
        degree_entropy = entropy(degree_probs[degree_probs > 0])
    else:
        degree_entropy = 0.0
    
    # Centrality measures (on undirected version for simplicity)
    try:
        betweenness = list(nx.betweenness_centrality(G_undirected).values())
        closeness = list(nx.closeness_centrality(G_undirected).values())
        pagerank = list(nx.pagerank(G).values())
    except:
        betweenness = [0]
        closeness = [0]
        pagerank = [0]
    
    # Connected components
    n_weak_components = nx.number_weakly_connected_components(G)
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    largest_cc_ratio = len(largest_cc) / n_nodes if n_nodes > 0 else 0
    
    # Average shortest path (only if connected and small enough)
    try:
        if nx.is_connected(G_undirected):
            avg_shortest_path = nx.average_shortest_path_length(G_undirected)
        else:
            # Use largest component
            largest_subgraph = G_undirected.subgraph(largest_cc)
            avg_shortest_path = nx.average_shortest_path_length(largest_subgraph)
    except:
        avg_shortest_path = np.nan
    
    features = {
        'user_id': user_id,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'avg_degree': np.mean(degrees),
        'max_degree': np.max(degrees),
        'std_degree': np.std(degrees),
        'avg_weighted_degree': np.mean(weighted_degrees),
        'max_weighted_degree': np.max(weighted_degrees),
        'avg_clustering': avg_clustering,
        'transitivity': transitivity,
        'degree_assortativity': degree_assortativity,
        'weight_entropy': weight_entropy,
        'degree_entropy': degree_entropy,
        'betweenness_mean': np.mean(betweenness),
        'betweenness_max': np.max(betweenness),
        'closeness_mean': np.mean(closeness),
        'closeness_max': np.max(closeness),
        'pagerank_mean': np.mean(pagerank),
        'pagerank_max': np.max(pagerank),
        'n_weak_components': n_weak_components,
        'largest_cc_ratio': largest_cc_ratio,
        'avg_shortest_path': avg_shortest_path,
    }
    
    all_features.append(features)
    print(f"  {user_id}: {n_nodes} nodes, {n_edges} edges")

df = pd.DataFrame(all_features)
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} rows to {output_path}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nSummary statistics:")
print(df.describe().to_string())