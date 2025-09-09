import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

# === CONFIGURATION ===
MODEL_PATH = "fine_tuned_models/RoBERTa-Large/seed_3"
DATA_DIR = "lens_outputs/md_offense/RoBERTa-Large/fine_tuned/test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMBIGUITY_LEVELS = ["A0", "A+", "A++"] 

# === LOAD MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# === LOOP THROUGH AMBIGUITY LEVELS ===
for level in AMBIGUITY_LEVELS:
    meta_path = os.path.join(DATA_DIR, level, "1D_lenses", "raw_metadata.tsv")  
    if not os.path.exists(meta_path):
        print(f"Missing raw_metadata.tsv for {level}, skipping.")
        continue

    df = pd.read_csv(meta_path, sep="\t")
    texts = df["Text"].astype(str).tolist()

    predictions, conf_0, conf_1 = [], [], []

    for i in range(0, len(texts), 16):
        batch = texts[i:i+16]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)

        predictions.extend(preds)
        conf_0.extend(probs[:, 0])
        conf_1.extend(probs[:, 1])

    # Add predictions
    df["model_prediction"] = predictions
    df["model_conf_label0"] = conf_0
    df["model_conf_label1"] = conf_1

    # Overwrite the metadata file in lens folder
    df.to_csv(meta_path, sep="\t", index=False)
    print(f"Updated lens metadata with predictions for {level}")