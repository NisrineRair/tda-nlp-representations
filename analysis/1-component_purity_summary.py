import os
import json
import networkx as nx
import pandas as pd
from collections import Counter

# === CONFIG ===
LABEL_TYPE = "Offensive_binary_label"  # or "model_prediction"

CONFIG = {
    "MAPPER_DIR": "mapper/mapper_visualizations_chosen_hypermapper",
    "OUTPUT_DIR": "analysis/results_excel",
    "SUBSETS": ["test"],
    "PARAMS": ["hdbscan_final_cubes40_overlap3"],
    "MODELS": ["RoBERTa-Large"],
    "VERSIONS": ["base", "fine_tuned"],
    "LENSES": ["cosine_centroid_1d", "eccentricity_1d", "l2norm_1d", "pca_1d", "random1_1d", "random2_1d"]
}

# === Dynamic output filename
if LABEL_TYPE == "Offensive_binary_label":
    OUTPUT_NAME = "component_level_summary_test_offensive_majority.xlsx"
else:
    OUTPUT_NAME = "component_level_summary_test_model_prediction.xlsx"

OUTPUT_PATH = os.path.join(CONFIG["OUTPUT_DIR"], OUTPUT_NAME)

# === Compute node purity ===
def node_purity(points):
    counts = Counter(p[LABEL_TYPE] for p in points if LABEL_TYPE in p)
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0, 0, 0
    majority = max(counts.values())
    purity = majority / total
    return purity, total, counts.get(0, 0), counts.get(1, 0)

# === Traverse all mapper_graph.json files ===
all_graphs = []
for root, _, files in os.walk(CONFIG["MAPPER_DIR"]):
    if "mapper_graph.json" in files:
        all_graphs.append(os.path.join(root, "mapper_graph.json"))

component_records = []

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
    for node_id in graph["nodes"]:
        G.add_node(node_id)
    for src, tgt_list in graph.get("links", {}).items():
        for tgt in tgt_list:
            if src in G.nodes and tgt in G.nodes:
                G.add_edge(src, tgt)

    for comp_id, component in enumerate(nx.connected_components(G)):
        node_majority_labels = []
        total_points = 0
        total_label_counter = Counter()
        node_purities = []
        node_sizes = []

        for node_id in component:
            node = graph["nodes"][node_id]
            points = node["points"]
            p, size, l0, l1 = node_purity(points)
            if size == 0:
                continue
            node_majority = 0 if l0 >= l1 else 1
            node_majority_labels.append(node_majority)
            node_purities.append((p, size))
            node_sizes.append(size)
            total_points += size
            total_label_counter[0] += l0
            total_label_counter[1] += l1

        if total_points == 0:
            continue

        avg_node_purity = round(sum(p for p, _ in node_purities) / len(node_purities), 4)
        weighted_node_purity = round(sum(p * s for p, s in node_purities) / total_points, 4)
        majority_label = 0 if total_label_counter[0] >= total_label_counter[1] else 1
        component_purity = round(max(total_label_counter.values()) / total_points, 4)
        majority_label_nodewise = Counter(node_majority_labels).most_common(1)[0][0]

        component_records.append({
            "model": model,
            "version": version,
            "subset": subset,
            "ambiguity": ambiguity,
            "lens": lens,
            "param": param,
            "json_path": path,
            "component_id": comp_id,
            "num_nodes": len(component),
            "total_points": total_points,
            "label_0_count": total_label_counter[0],
            "label_1_count": total_label_counter[1],
            "majority_label": majority_label,
            "majority_label_nodewise": majority_label_nodewise,
            "avg_node_purity": avg_node_purity,
            "weighted_node_purity": weighted_node_purity,
            "component_purity": component_purity
        })

# === Save to Excel ===
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
df = pd.DataFrame(component_records)
df.to_excel(OUTPUT_PATH, index=False)
print(f"Saved {len(df)} component rows to: {OUTPUT_PATH}")
