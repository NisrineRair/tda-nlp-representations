import os
import pandas as pd
import numpy as np
import kmapper as km
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from hdbscan import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm

# === CONFIG ===
MODEL_NAME = "RoBERTa-Large"
VERSION = "fine_tuned"
SUBSET = "test"
AMBIGUITY_LEVEL = "A++"  
LENS_NAME = "pca_1d.tsv"
LENS_DIM_FOLDER = "1D_lenses"
EMBEDDINGS_DIR = "embeddings/Fine_tuned_embeddings/md_offense"
LENS_OUTPUT_DIR = "lens_outputs/md_offense"
MAPPER_OUTPUT_DIR = "scripts/mapper_hparam_search/mapper_output/test_clustering"

N_CUBES = 40
OVERLAP = 0.3
SEED = 42

# Define clusterers
CLUSTERERS = {
    "kmeans": KMeans(n_clusters=2, random_state=SEED),
    "dbscan": DBSCAN(eps=0.5, min_samples=1),
    "agglomerative": AgglomerativeClustering(n_clusters=2),
    "hdbscan": HDBSCAN(min_cluster_size=2, min_samples=1),
}

# Paths
lens_path = os.path.join(LENS_OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, AMBIGUITY_LEVEL, LENS_DIM_FOLDER, LENS_NAME)
meta_path = os.path.join(LENS_OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, AMBIGUITY_LEVEL, LENS_DIM_FOLDER, "raw_metadata.tsv")
embeddings_path = os.path.join(EMBEDDINGS_DIR, MODEL_NAME, "seed_3", SUBSET, AMBIGUITY_LEVEL, "cls_embeddings.tsv")

# Check files
if not all(os.path.exists(p) for p in [lens_path, meta_path, embeddings_path]):
    raise FileNotFoundError("Some input files are missing.")

# Load data
lens = pd.read_csv(lens_path, sep="\t").values
meta = pd.read_csv(meta_path, sep="\t")
embeddings = pd.read_csv(embeddings_path, sep="\t", header=None).values

# Normalize lens
lens_scaled = MinMaxScaler().fit_transform(lens)

# Setup coloring
color_columns = ["Offensive_binary_label", "Agreement_level", "Domain"]
if VERSION == "fine_tuned":
    color_columns += ["model_prediction", "model_conf_label0", "model_conf_label1"]

color_value_dict = {}
original_labels = {}

for col in color_columns:
    if meta[col].dtype == "object":
        meta[col], uniques = pd.factorize(meta[col])
        original_labels[col] = list(uniques)
    else:
        original_labels[col] = sorted(meta[col].unique().astype(str))
    color_value_dict[col] = meta[col].values

# === TEST LOOP ===
for clusterer_name, clusterer in CLUSTERERS.items():
    print(f"Running Mapper with {clusterer_name}...")

    mapper = km.KeplerMapper(verbose=0)

    graph = mapper.map(
        X=embeddings,
        lens=lens_scaled,
        cover=km.Cover(n_cubes=N_CUBES, perc_overlap=OVERLAP),
        clusterer=clusterer,
    )

    # Output path
    clusterer_folder = f"{clusterer_name}"
    out_path = os.path.join(
        MAPPER_OUTPUT_DIR,
        MODEL_NAME,
        VERSION,
        SUBSET,
        AMBIGUITY_LEVEL,
        LENS_DIM_FOLDER,
        os.path.splitext(LENS_NAME)[0],
        clusterer_folder
    )
    os.makedirs(out_path, exist_ok=True)

    mapper.visualize(
        graph,
        path_html=os.path.join(out_path, "mapper.html"),
        title=f"Mapper | {MODEL_NAME} | {LENS_NAME} | {clusterer_name}",
        color_values=pd.DataFrame(color_value_dict),
        color_function_name=color_columns,
        node_color_function=["mean", "std", "median", "min", "max"],
    )

    # Save legend
    legend_lines = ["Mapper Color Legend (viridis)\n"]
    for col_name, labels in original_labels.items():
        legend_lines.append(f"{col_name}:")
        for i, label in enumerate(labels):
            r, g, b = cm.viridis(i / max(1, len(labels) - 1))[:3]
            rgb = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            legend_lines.append(f"  {i} = {label} -> {rgb}")
        legend_lines.append("")

    with open(os.path.join(out_path, "mapper_color_legend.txt"), "w") as f:
        f.write("\n".join(legend_lines))

    print(f"Saved: {out_path}/mapper.html")
