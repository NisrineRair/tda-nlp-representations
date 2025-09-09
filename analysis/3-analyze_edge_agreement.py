import os
import json
import networkx as nx
import pandas as pd
from collections import Counter

# === CONFIG ===
CONFIG = {
    "MAPPER_DIR": "mapper/mapper_visualizations_chosen_hypermapper",
    "OUTPUT_CSV": "analysis/results_excel/edge_agreement_summary.xlsx",
    "SUBSETS": ["test"],
    "PARAMS": ["hdbscan_final_cubes40_overlap3"],
    "MODELS": ["RoBERTa-Large"],
    "VERSIONS": ["base", "fine_tuned"],
    "LENSES": ["cosine_centroid_1d", "eccentricity_1d", "l2norm_1d", "pca_1d", "random1_1d", "random2_1d"] 
}

# # === Helper: majority label for a node ===
def majority_label(points):
    label_counts = Counter(p["Offensive_binary_label"] for p in points if "Offensive_binary_label" in p)
    if not label_counts:
        return None, 0
    maj = label_counts.most_common(1)[0][0]
    size = sum(label_counts.values())
    return maj, size

# === Traverse mapper graphs ===
all_graphs = []
for root, _, files in os.walk(CONFIG["MAPPER_DIR"]):
    if "mapper_graph.json" in files:
        all_graphs.append(os.path.join(root, "mapper_graph.json"))

records = []

for path in all_graphs:
    with open(path, "r") as f:
        graph = json.load(f)

    parts = path.replace(CONFIG["MAPPER_DIR"], "").strip(os.sep).split(os.sep)
    model, version, subset, ambiguity, lens, param = (parts + [""] * 6)[:6]

    if CONFIG["SUBSETS"] and subset not in CONFIG["SUBSETS"]:
        continue
    if CONFIG["PARAMS"] and param not in CONFIG["PARAMS"]:
        continue
    if CONFIG["MODELS"] and model not in CONFIG["MODELS"]:
        continue
    if CONFIG["VERSIONS"] and version not in CONFIG["VERSIONS"]:
        continue
    if CONFIG["LENSES"] and lens not in CONFIG["LENSES"]:
        continue

    G = nx.Graph()
    node_labels = {}
    node_sizes = {}

    for node_id, node in graph["nodes"].items():
        maj_label, size = majority_label(node["points"])
        if maj_label is not None:
            G.add_node(node_id)
            node_labels[node_id] = maj_label
            node_sizes[node_id] = size

    for src, tgt_list in graph.get("links", {}).items():
        for tgt in tgt_list:
            if src in G.nodes and tgt in G.nodes:
                G.add_edge(src, tgt)

    total_edges = 0
    label_agree_edges = 0
    weighted_agree = 0
    total_weight = 0

    for u, v in G.edges():
        total_edges += 1
        w = node_sizes[u] + node_sizes[v]
        total_weight += w
        if node_labels[u] == node_labels[v]:
            label_agree_edges += 1
            weighted_agree += w

    edge_agreement_unweighted = label_agree_edges / total_edges if total_edges > 0 else 0.0
    edge_agreement_weighted = weighted_agree / total_weight if total_weight > 0 else 0.0

    records.append({
        "model": model,
        "version": version,
        "subset": subset,
        "ambiguity": ambiguity,
        "lens": lens,
        "param": param,
        "edge_agreement_unweighted": round(edge_agreement_unweighted, 4),
        "edge_agreement_weighted": round(edge_agreement_weighted, 4),
        "num_nodes": len(G.nodes),
        "num_edges": len(G.edges)
    })

# Save results
summary_df = pd.DataFrame(records)
summary_df.to_excel(CONFIG["OUTPUT_CSV"], index=False)
print(f"Saved global graph metrics for {len(summary_df)} graphs to: {CONFIG['OUTPUT_CSV']}")
