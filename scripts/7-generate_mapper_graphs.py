import os
import json
import pandas as pd
import numpy as np
import kmapper as km
from hdbscan import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import cm

# === CONFIG ===
MODEL_NAME =  "RoBERTa-Large" 
VERSION = "fine_tuned" #fine_tuned
SUBSET = "test"
AMBIGUITY_LEVELS = [ "A0", "A+", "A++"]  # List of ambiguity levels
LENS_DIM_FOLDERS = ["1D_lenses"]  # Loop through both 1D and 2D lenses
EMBEDDINGS_DIR_BASE = "embeddings/Base_embeddings/md_offense"
EMBEDDINGS_DIR_FINE_TUNED = "embeddings/Fine_tuned_embeddings/md_offense"
LENS_OUTPUT_DIR = "lens_outputs/md_offense"
MAPPER_OUTPUT_DIR = "mapper/mapper_visualizations_chosen_hypermapper"

# === Hyperparameters ===
HYPERPARAMS = {
    "1D_lenses": {"n_cubes": 40, "overlap": 0.3},
    # "2D_lenses": {"n_cubes": 20, "overlap": 0.2},
}
N_CUBES = HYPERPARAMS["1D_lenses"]["n_cubes"]  # Default value, can be changed dynamically
OVERLAP = HYPERPARAMS["1D_lenses"]["overlap"]  # Default value, can be changed dynamically



# === Function to get lens files from a folder for a specific ambiguity level ===
def get_lens_files_for_level(ambiguity_level, lens_dim_folder):
    folder_path = os.path.join(
        LENS_OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, ambiguity_level, lens_dim_folder
    )
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return []
    
    return [f for f in os.listdir(folder_path) if f.endswith(".tsv") and f != "raw_metadata.tsv"]


# === Function to process and visualize each lens and ambiguity level ===
def process_lens_and_ambiguity(lens_name, ambiguity_level, lens_dim_folder, embeddings_dir):
    # Update the N_CUBES and OVERLAP values dynamically based on lens_dim_folder
    global N_CUBES, OVERLAP
    N_CUBES = HYPERPARAMS[lens_dim_folder]["n_cubes"]
    OVERLAP = HYPERPARAMS[lens_dim_folder]["overlap"]
    
    # === Paths ===
    lens_path = os.path.join(LENS_OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, ambiguity_level, lens_dim_folder, lens_name)
    meta_path = os.path.join(LENS_OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, ambiguity_level, lens_dim_folder, "raw_metadata.tsv")
    if VERSION == "fine_tuned":
        embeddings_path = os.path.join(embeddings_dir, MODEL_NAME, "seed_3", SUBSET, ambiguity_level, "cls_embeddings.tsv")
    else:
        embeddings_path = os.path.join(embeddings_dir, MODEL_NAME, SUBSET, ambiguity_level, "cls_embeddings.tsv")

    if not all(os.path.exists(p) for p in [lens_path, meta_path, embeddings_path]):
        print(f"Missing input files for lens {lens_name} and ambiguity {ambiguity_level}. Skipping.")
        return

    # === Load data ===
    lens = pd.read_csv(lens_path, sep="\t").values
    meta = pd.read_csv(meta_path, sep="\t")
    embeddings = pd.read_csv(embeddings_path, sep="\t", header=None).values
    meta["id"] = meta.index

    # Normalize lens
    lens_scaled = MinMaxScaler().fit_transform(lens)

    # === Mapper ===
    mapper = km.KeplerMapper(verbose=1)
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)

    graph = mapper.map(
        X=embeddings,
        lens=lens_scaled,
        cover=km.Cover(n_cubes=N_CUBES, perc_overlap=OVERLAP),
        clusterer=clusterer,
    )

    # === Track included/excluded ===
    included_ids = set()
    for node_data in graph["nodes"].values():
        for idx in node_data:
            included_ids.add(idx)

    excluded_ids = sorted(list(set(meta.index) - included_ids))
    excluded_metadata = meta.loc[excluded_ids]

    # === Save paths ===
    lens_folder = os.path.splitext(lens_name)[0]
    param_folder = f"hdbscan_final_cubes{N_CUBES}_overlap{int(OVERLAP*10)}"
    out_path = os.path.join(MAPPER_OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, ambiguity_level, lens_folder, param_folder)
    os.makedirs(out_path, exist_ok=True)

    excluded_path = os.path.join(out_path, "excluded_points.tsv")
    excluded_metadata.to_csv(excluded_path, sep="\t", index=False)

    # === Coloring and Functions ===
    if VERSION == "base":
        color_columns = ["Offensive_binary_label", "Agreement_level", "Domain"]  # Use this for base model (no model_prediction)
    else:
        color_columns = ["Offensive_binary_label", "Agreement_level", "Domain", "model_prediction"]  # Use this for fine-tuned model

    color_value_dict = {}


    # Save original labels for legend extraction
    original_labels = {}

    for col in color_columns:
        if meta[col].dtype == "object":
            meta[col], uniques = pd.factorize(meta[col])
            original_labels[col] = list(uniques)
        else:
            original_labels[col] = sorted(meta[col].unique().astype(str))
        color_value_dict[col] = meta[col].values

    # === Visualize Mapper with color from metadata ===
    mapper.visualize(
        graph,
        path_html=os.path.join(out_path, "mapper.html"),
        title=f"Mapper | {VERSION} |{MODEL_NAME} | {lens_folder} | {ambiguity_level}| {N_CUBES}| {OVERLAP}",
        color_values=pd.DataFrame(color_value_dict),  # Directly using metadata for coloring
        color_function_name=color_columns,  # Dynamically selecting color function
        node_color_function=["mean","std","median","min","max"] ,  
    )

    # === Save color legend with RGB info ===
    legend_lines = ["Mapper Color Legend (colormap: viridis)\n"]
    cmap = cm.get_cmap("viridis")

    for col_name, label_list in original_labels.items():
        legend_lines.append(f"{col_name}:")
        values = np.linspace(0, 1, len(label_list))
        for i, v in enumerate(values):
            r, g, b = cmap(v)[:3]
            rgb_str = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            legend_lines.append(f"  {i} = {label_list[i]} -> {rgb_str}")
        legend_lines.append("")

    legend_txt = os.path.join(out_path, "mapper_color_legend.txt")
    with open(legend_txt, "w") as f:
        f.write("\n".join(legend_lines))

    # === Enrich and save graph JSON ===
    for node_name in graph["nodes"]:
        point_indices = graph["nodes"][node_name]
        full_metadata = meta.iloc[point_indices].to_dict(orient="records")
        graph["nodes"][node_name] = {"cluster_id": node_name, "points": full_metadata}

    with open(os.path.join(out_path, "mapper_graph.json"), "w") as f:
        json.dump(graph, f, indent=2)

    # === Final report ===
    print(f"\n Saved HTML, JSON, and color legend to {out_path}")
    print(f"Samples in graph: {len(included_ids)}")
    print(f" Excluded: {len(excluded_ids)}")
    print(f" Color legend: {legend_txt}")

# === Loop over all lens folders, ambiguity levels, and lens files ===
for lens_dim_folder in LENS_DIM_FOLDERS:
    for ambiguity_level in AMBIGUITY_LEVELS:
        lens_files = get_lens_files_for_level(ambiguity_level, lens_dim_folder)
        for lens_name in lens_files:
            embeddings_dir = (
                EMBEDDINGS_DIR_FINE_TUNED if VERSION == "fine_tuned" else EMBEDDINGS_DIR_BASE
            )
            process_lens_and_ambiguity(lens_name, ambiguity_level, lens_dim_folder, embeddings_dir)
