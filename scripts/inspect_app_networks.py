import os
import networkx as nx

folder = r"C:\doorway_bench\data\raw\networkdata\graphs"

# List all GML files
gml_files = sorted([f for f in os.listdir(folder) if f.endswith('.gml')])
print(f"Found {len(gml_files)} GML files")
print(f"First 5 filenames: {gml_files[:5]}")
print(f"Last 5 filenames: {gml_files[-5:]}")

# Load the first one
first_file = os.path.join(folder, gml_files[0])
print(f"\n{'='*60}")
print(f"INSPECTING: {gml_files[0]}")
print(f"{'='*60}")

G = nx.read_gml(first_file)

print(f"\nNumber of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Is directed: {G.is_directed()}")
print(f"Is weighted: {nx.is_weighted(G)}")

# Show node attributes
print(f"\nNode attributes (first 5 nodes):")
for node, attrs in list(G.nodes(data=True))[:5]:
    print(f"  {node}: {attrs}")

# Show edge attributes
print(f"\nEdge attributes (first 5 edges):")
for u, v, attrs in list(G.edges(data=True))[:5]:
    print(f"  {u} -> {v}: {attrs}")

# Show all unique node names (apps)
print(f"\nAll nodes (apps): {list(G.nodes())[:20]}")

# Check if there are more files with different structures
if len(gml_files) > 1:
    second_file = os.path.join(folder, gml_files[1])
    G2 = nx.read_gml(second_file)
    print(f"\n{'='*60}")
    print(f"INSPECTING: {gml_files[1]}")
    print(f"{'='*60}")
    print(f"Nodes: {G2.number_of_nodes()}, Edges: {G2.number_of_edges()}")
    print(f"Sample nodes: {list(G2.nodes())[:10]}")