import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

# === CONFIG ===
MODEL_NAME = "RoBERTa-Large"
VERSION = "fine_tuned"  # or "fine_tuned"
SUBSET = "test"  # train / dev / test
AMBIGUITY_LEVELS = ["A0", "A+", "A++"]

BASE_EMBEDDINGS_DIR = "embeddings/Base_embeddings/md_offense"
FINE_TUNED_EMBEDDINGS_DIR = "embeddings/Fine_tuned_embeddings/md_offense"

OUTPUT_DIR = "lens_outputs/md_offense"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FUNCTIONS ===
def compute_pca_1d(train_embeddings, test_embeddings):
    pca = PCA(n_components=1)
    pca.fit(train_embeddings)
    return pca.transform(test_embeddings)

def compute_two_random_1d(test_embeddings, random_state=42):
    np.random.seed(random_state)
    random_dir_1 = np.random.randn(test_embeddings.shape[1])
    random_dir_1 /= np.linalg.norm(random_dir_1)
    proj_1 = np.dot(test_embeddings, random_dir_1).reshape(-1, 1)

    random_dir_2 = np.random.randn(test_embeddings.shape[1])
    random_dir_2 /= np.linalg.norm(random_dir_2)
    proj_2 = np.dot(test_embeddings, random_dir_2).reshape(-1, 1)

    return proj_1, proj_2

def compute_cosine_centroid_1d(train_embeddings, train_labels, test_embeddings):
    labels = np.array(train_labels)
    c0 = train_embeddings[labels == 0].mean(axis=0)
    c1 = train_embeddings[labels == 1].mean(axis=0)
    centroid_vector = c1 - c0
    centroid_vector /= np.linalg.norm(centroid_vector)
    return np.dot(test_embeddings, centroid_vector).reshape(-1, 1)

def compute_l2_norm_1d(test_embeddings):
    norms = np.linalg.norm(test_embeddings, axis=1)
    return norms.reshape(-1, 1)

def compute_eccentricity_1d(train_embeddings, test_embeddings):
    distances = cosine_distances(test_embeddings, train_embeddings)
    eccentricity = np.max(distances, axis=1)
    return eccentricity.reshape(-1, 1)

# === MAIN LOOP ===
for ambiguity in AMBIGUITY_LEVELS:

    if VERSION == "base":
        train_embed_folder = os.path.join(BASE_EMBEDDINGS_DIR, MODEL_NAME, "train", ambiguity)
        test_embed_folder = os.path.join(BASE_EMBEDDINGS_DIR, MODEL_NAME, SUBSET, ambiguity)
    else:
        train_embed_folder = os.path.join(FINE_TUNED_EMBEDDINGS_DIR, MODEL_NAME,"seed_3", "train", ambiguity)
        test_embed_folder = os.path.join(FINE_TUNED_EMBEDDINGS_DIR, MODEL_NAME,"seed_3", SUBSET, ambiguity)
    # Paths
    train_embed_path = os.path.join(train_embed_folder, "cls_embeddings.tsv")
    train_meta_path = os.path.join(train_embed_folder, "raw_metadata.tsv")
    test_embed_path = os.path.join(test_embed_folder, "cls_embeddings.tsv")
    test_meta_path = os.path.join(test_embed_folder, "raw_metadata.tsv")

    if not (os.path.exists(train_embed_path) and os.path.exists(test_embed_path)):
        print(f"Missing embeddings for ambiguity {ambiguity}")
        continue

    # Load
    train_embeddings = pd.read_csv(train_embed_path, sep="\t", header=None).values
    test_embeddings = pd.read_csv(test_embed_path, sep="\t", header=None).values
    train_metadata = pd.read_csv(train_meta_path, sep="\t")
    test_metadata = pd.read_csv(test_meta_path, sep="\t")

    train_labels = train_metadata["Offensive_binary_label"].values

    # === Compute lenses ===
    pca_1d = compute_pca_1d(train_embeddings, test_embeddings)
    random1_1d, random2_1d = compute_two_random_1d(test_embeddings)
    centroid_1d = compute_cosine_centroid_1d(train_embeddings, train_labels, test_embeddings)
    l2norm_1d = compute_l2_norm_1d(test_embeddings)
    eccentricity_1d = compute_eccentricity_1d(train_embeddings, test_embeddings)

    # === Save lenses ===
    lens_output_folder = os.path.join(OUTPUT_DIR, MODEL_NAME, VERSION, SUBSET, ambiguity, "1D_lenses")
    os.makedirs(lens_output_folder, exist_ok=True)

    pd.DataFrame(pca_1d, columns=["pca_1d"]).to_csv(os.path.join(lens_output_folder, "pca_1d.tsv"), sep="\t", index=False)
    pd.DataFrame(random1_1d, columns=["random1_1d"]).to_csv(os.path.join(lens_output_folder, "random1_1d.tsv"), sep="\t", index=False)
    pd.DataFrame(random2_1d, columns=["random2_1d"]).to_csv(os.path.join(lens_output_folder, "random2_1d.tsv"), sep="\t", index=False)
    pd.DataFrame(centroid_1d, columns=["cosine_centroid_1d"]).to_csv(os.path.join(lens_output_folder, "cosine_centroid_1d.tsv"), sep="\t", index=False)
    pd.DataFrame(l2norm_1d, columns=["l2norm_1d"]).to_csv(os.path.join(lens_output_folder, "l2norm_1d.tsv"), sep="\t", index=False)
    pd.DataFrame(eccentricity_1d, columns=["eccentricity_1d"]).to_csv(os.path.join(lens_output_folder, "eccentricity_1d.tsv"), sep="\t", index=False)

    # Also copy metadata for Mapper visualization
    test_metadata.to_csv(os.path.join(lens_output_folder, "raw_metadata.tsv"), sep="\t", index=False)

    print(f"Saved lenses for {ambiguity}")

