import os
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer

# Configurations
class Config:
    EMBEDDINGS_OUTPUT_FOLDER = "embeddings/Fine_tuned_embeddings"
    MODEL_NAMES = {
        # "RoBERTa-Large": "FacebookAI/roberta-large"
        "BERT-Base": "bert-base-uncased",
        "DistilBERT": "distilbert/distilbert-base-uncased",
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "E5-Base": "intfloat/e5-base",
        "DeBERTa-Base": "microsoft/deberta-base",
      
    }
    BASE_PATHS = {
        "md_offense": "data/processed_data/md_offense"
    }
    FINE_TUNED_MODELS_DIR = "fine_tuned_models"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16

# Mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

#  CLS token
def get_cls_embedding(model_output):
    return model_output.last_hidden_state[:, 0, :]

# Find best checkpoint based on eval_f1_macro
def find_best_checkpoint(seed_path):
    eval_log_path = os.path.join(seed_path, "logs", "eval_results.csv")
    if not os.path.exists(eval_log_path):
        print(f"No eval.csv found in {eval_log_path}")
        return None
    df = pd.read_csv(eval_log_path)
    if "eval_f1_macro" not in df.columns or "global_step" not in df.columns:
        print("Missing columns in eval.csv")
        return None
    best_row = df.loc[df["eval_f1_macro"].idxmax()]
    best_step = int(best_row["global_step"])
    best_checkpoint = os.path.join(seed_path, f"checkpoint-{best_step}")
    return best_checkpoint if os.path.exists(best_checkpoint) else None

# Get subfolder structure (train/dev/test -> A0/A+/A++)
def get_subfolders(dataset_name):
    base_path = Config.BASE_PATHS[dataset_name]
    structure = {}
    for split in ["train", "dev", "test"]:
        path = os.path.join(base_path, split)
        if os.path.exists(path):
            structure[split] = [s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
    return structure

# Compute embeddings for a list of texts
def get_hf_embeddings(sentences, model, tokenizer):
    model.eval()
    mean_all, cls_all = [], []
    for i in range(0, len(sentences), Config.BATCH_SIZE):
        batch = sentences[i:i+Config.BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(Config.DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        mean_all.extend(mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy())
        cls_all.extend(get_cls_embedding(outputs).cpu().numpy())
    return mean_all, cls_all

# Save embeddings and metadata
def save_embeddings(mean_emb, cls_emb, dataset, dataset_name, model_name, seed_name, split, ambiguity):
    out_dir = os.path.join(Config.EMBEDDINGS_OUTPUT_FOLDER, dataset_name, model_name, seed_name, split, ambiguity)
    os.makedirs(out_dir, exist_ok=True)

    # Save embeddings
    pd.DataFrame(mean_emb).to_csv(os.path.join(out_dir, "mean_embeddings.tsv"), sep="\t", index=False, header=False)
    pd.DataFrame(cls_emb).to_csv(os.path.join(out_dir, "cls_embeddings.tsv"), sep="\t", index=False, header=False)

    # Save raw metadata (all columns)
    raw_metadata_df = pd.DataFrame(dataset)
    raw_metadata_df.to_csv(os.path.join(out_dir, "raw_metadata.tsv"), sep="\t", index=False)

    print(f"Saved: {split}/{ambiguity} | {len(mean_emb)} embeddings | Metadata & Raw Metadata")

# Main 
def main():
    structure = get_subfolders("md_offense")
    for model_name in Config.MODEL_NAMES:
        model_dir = os.path.join(Config.FINE_TUNED_MODELS_DIR, model_name)
        if not os.path.isdir(model_dir): continue

        for seed_name in os.listdir(model_dir):
            seed_path = os.path.join(model_dir, seed_name)
            best_checkpoint = find_best_checkpoint(seed_path)
            if not best_checkpoint:
                print(f" No checkpoint for {model_name} | {seed_name}")
                continue

            print(f"\n {model_name} | {seed_name} | Checkpoint: {best_checkpoint}")
            tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
            model = AutoModel.from_pretrained(best_checkpoint).to(Config.DEVICE)

            for split, ambiguities in structure.items():
                for ambiguity in ambiguities:
                    ds_path = os.path.join(Config.BASE_PATHS["md_offense"], split, ambiguity)
                    if not os.path.exists(ds_path): continue
                    dataset = load_from_disk(ds_path)

                    mean_emb, cls_emb = get_hf_embeddings(dataset["Text"], model, tokenizer)
                    save_embeddings(mean_emb, cls_emb, dataset, "md_offense", model_name.replace("/", "_"), seed_name, split, ambiguity)

if __name__ == "__main__":
    main()
