import os
import pandas as pd
import numpy as np
import kmapper as km
from hdbscan import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_distances
from matplotlib import cm

# === CONFIG ===
MODEL_NAME = "RoBERTa-Large"
VERSION = "fine_tuned"
SUBSET = "test"
AMBIGUITY_LEVEL = "A++"
LENS_NAME = "random1_1d.tsv"
LENS_DIM_FOLDER = "1D_lenses"
EMBEDDINGS_DIR = "embeddings/Fine_tuned_embeddings/md_offense"
LENS_OUTPUT_DIR = "lens_outputs/md_offense"

N_CUBES_LIST = [40]  # You can vary this too to test n_cube effect
OVERLAP_LIST = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
SEED = 42

# === PATHS ===
lens_path = os.path.join(LENS_OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, AMBIGUITY_LEVEL, LENS_DIM_FOLDER, LENS_NAME)
meta_path = os.path.join(LENS_OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, AMBIGUITY_LEVEL, LENS_DIM_FOLDER, "raw_metadata.tsv")
embeddings_path = os.path.join(EMBEDDINGS_DIR, MODEL_NAME, "seed_3", SUBSET, AMBIGUITY_LEVEL, "cls_embeddings.tsv")

if not all(os.path.exists(p) for p in [lens_path, meta_path, embeddings_path]):
    raise FileNotFoundError("Some input files are missing.")

# === LOAD ===
lens = pd.read_csv(lens_path, sep="\t").values
meta = pd.read_csv(meta_path, sep="\t")
embeddings = pd.read_csv(embeddings_path, sep="\t", header=None).values
lens_scaled = MinMaxScaler().fit_transform(lens)

# === LOOP ===
for n_cubes in N_CUBES_LIST:
    for overlap in OVERLAP_LIST:
        print(f"🔍 Mapper | cubes={n_cubes} | overlap={overlap}")

        mapper = km.KeplerMapper(verbose=0)
        clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)

        graph = mapper.map(
            X=embeddings,
            lens=lens_scaled,
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap),
            clusterer=clusterer,
        )

     
        # === Automatically determine test type based on N_CUBES_LIST and OVERLAP_LIST
        if len(N_CUBES_LIST) > 1 and len(OVERLAP_LIST) == 1:
            test_type = "test_n_cubes"
        elif len(OVERLAP_LIST) > 1 and len(N_CUBES_LIST) == 1:
            test_type = "test_overlap"
        else:
            test_type = "test_combined"
        lens_folder = os.path.splitext(LENS_NAME)[0]
        param_folder = f"hdbscan_cubes{n_cubes}_overlap{int(overlap*10)}"
        test_root_folder = f"scripts/mapper_hparam_search/mapper_output/{test_type}"

        out_path = os.path.join(test_root_folder, lens_folder, param_folder)
        os.makedirs(out_path, exist_ok=True)


        # === Color Columns
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

        mapper.visualize(
            graph,
            path_html=os.path.join(out_path, "mapper.html"),
            title=f"Mapper | {MODEL_NAME} | {lens_folder} | cubes={n_cubes} overlap={overlap}",
            color_values=pd.DataFrame(color_value_dict),
            color_function_name=color_columns,
            node_color_function=["mean", "std", "median", "min", "max"],
        )

        # === Save Legend
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

        print(f"Saved to: {out_path}")
