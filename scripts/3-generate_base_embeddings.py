import os
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

# Configuration: Model selection and embeddings output folder
class Config:
    EMBEDDINGS_OUTPUT_FOLDER = "embeddings/Base_embeddings"
    MODEL_NAMES = {
        # "RoBERTa-Large": "FacebookAI/roberta-large",
        "BERT-Base": "bert-base-uncased",
        "DistilBERT": "distilbert/distilbert-base-uncased",
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "E5-Base": "intfloat/e5-base",
        "DeBERTa-Base": "microsoft/deberta-base",
        
    }

    BASE_PATHS = {
        "md_offense": "data/processed_data/md_offense"
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16  # Batch processing for efficiency

# Mean Pooling Function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask  # Compute mean

# CLS Token Extraction From Last Hidden State
def get_cls_embedding(model_output):
    return model_output.last_hidden_state[:, 0, :]  # Take CLS token

# Function to compute embeddings using Hugging Face models
def get_hf_embeddings(sentences, model_name, model_path):
    """Fetch embeddings using Hugging Face transformers (batch processing)."""
    
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(Config.DEVICE)
    model.eval()  # Set to evaluation mode to disable dropout

    all_mean_embeddings = []
    all_cls_embeddings = []

    for i in range(0, len(sentences), Config.BATCH_SIZE):
        batch_sentences = sentences[i : i + Config.BATCH_SIZE]

        # 🔹 MD-Offense dataset (single sentence)
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)

        inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Get mean-pooling embeddings
        mean_embeddings = mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()
        # Get CLS token embeddings
        cls_embeddings = get_cls_embedding(outputs).cpu().numpy()

        all_mean_embeddings.extend(mean_embeddings)
        all_cls_embeddings.extend(cls_embeddings)

    print(f" Processed {len(all_mean_embeddings)}/{len(sentences)} embeddings for {model_name}")
    
    return all_mean_embeddings, all_cls_embeddings

# Function to save embeddings
def save_embeddings(mean_embeddings, cls_embeddings, dataset_name, model_name, subfolder_name, sub_subfolder):
    """Save both mean-pooling and CLS embeddings to TSV files."""
    output_dir = os.path.join(Config.EMBEDDINGS_OUTPUT_FOLDER, dataset_name, model_name.replace("/", "_"), subfolder_name, sub_subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mean-pooling embeddings
    mean_emb_df = pd.DataFrame(mean_embeddings)
    mean_emb_df.to_csv(os.path.join(output_dir, "mean_embeddings.tsv"), sep="\t", index=False, header=False)
    
    # Save CLS token embeddings
    cls_emb_df = pd.DataFrame(cls_embeddings)
    cls_emb_df.to_csv(os.path.join(output_dir, "cls_embeddings.tsv"), sep="\t", index=False, header=False)


# Function to save metadata
def save_metadata(sentences, labels, dataset_name, model_name, subfolder_name, sub_subfolder, raw_metadata):
    """Save metadata (single column for MD-Offense) and raw metadata."""
    output_dir = os.path.join(Config.EMBEDDINGS_OUTPUT_FOLDER, dataset_name, model_name.replace("/", "_"), subfolder_name, sub_subfolder)
    os.makedirs(output_dir, exist_ok=True)
    raw_metadata_file = os.path.join(output_dir, "raw_metadata.tsv")
    raw_metadata.to_csv(raw_metadata_file, sep="\t", index=False)

    print(f" Raw metadata saved: {len(sentences)} lines written to {raw_metadata_file}")

# Function to detect subfolders for MD-Offense
def get_subfolders(dataset_name):
    """Automatically detect subfolders (like A+, A++, A0) inside dataset directories."""
    base_path = Config.BASE_PATHS[dataset_name]
    subfolder_structure = {}

    if os.path.exists(base_path):
        # For each folder ('train', 'test', 'dev'), detect subfolders
        for folder in ['train', 'test', 'dev']:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                # List subfolders under 'train', 'test', 'dev'
                subfolder_structure[folder] = [subfolder for subfolder in os.listdir(folder_path) 
                                               if os.path.isdir(os.path.join(folder_path, subfolder))]
            else:
                subfolder_structure[folder] = []

    return subfolder_structure

def load_and_process_dataset(dataset_name, subfolder_name, model_name, model_path):
    """Load dataset, extract sentences, process embeddings, and save."""
    base_path = Config.BASE_PATHS[dataset_name]
    subfolder_path = os.path.join(base_path, subfolder_name)

    # Get the actual subfolders dynamically
    subfolder_structure = get_subfolders(dataset_name)

    # Check if the subfolder exists
    if subfolder_name in subfolder_structure:
        sub_subfolders = subfolder_structure[subfolder_name]
    else:
        print(f" No subfolders found in {subfolder_name}. Skipping.")
        return

    for sub_subfolder in sub_subfolders:
        # Construct full path for each subfolder (A+, A++, A0)
        dataset_path = os.path.join(subfolder_path, sub_subfolder)
        
        if not os.path.exists(dataset_path):
            print(f" Subfolder {sub_subfolder} not found in {subfolder_name}. Skipping.")
            continue

        # Load the dataset from the subfolder (A+, A++, A0)
        try:
            dataset = load_from_disk(dataset_path)
        except FileNotFoundError as e:
            print(f"Error loading dataset from {dataset_path}: {e}")
            continue

        # If dataset is a DatasetDict, select the correct split (train, dev, test)
        if isinstance(dataset, dict):  
            dataset = dataset[next(iter(dataset))]  # Use first split, like "train"

        # Check dataset structure and column names
        print("Dataset columns:", dataset.column_names)

        # Extract sentences & labels
        sentences = dataset["Text"]  # Ensure column name 'Text' is correct
        labels = dataset["Offensive_binary_label"]  # Ensure column name 'Offensive_binary_label' is correct

        # Create the raw metadata dataframe (get all columns)
        raw_metadata = dataset  # Get all columns

        print(f" Processing {len(sentences)} samples from {dataset_name} ({subfolder_name}/{sub_subfolder}) using {model_name}...")

        # Compute embeddings
        mean_embeddings, cls_embeddings = get_hf_embeddings(sentences, model_name, model_path)

        # Save embeddings and metadata
        save_embeddings(mean_embeddings, cls_embeddings, dataset_name, model_name, subfolder_name, sub_subfolder)
 

        save_metadata(sentences, labels, dataset_name, model_name, subfolder_name, sub_subfolder, raw_metadata)

        # Free GPU memory after processing each dataset
        torch.cuda.empty_cache()



# Main function to process all datasets and models
def main():
    for model_name, model_path in Config.MODEL_NAMES.items():
        print(f"\n Processing MD-Offense datasets with model: {model_name}...")

        # Loop through subfolders: train, test, dev
        subfolder_structure = get_subfolders("md_offense")

        for subfolder_name, sub_subfolders in subfolder_structure.items():
            for sub_subfolder in sub_subfolders:
                load_and_process_dataset("md_offense", subfolder_name, model_name, model_path)

if __name__ == "__main__":
    main()
