import os
import random
import json
import torch
import numpy as np
import csv
from datasets import load_from_disk, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback, TrainerCallback
)
from sklearn.metrics import accuracy_score, f1_score

# Configuration - modify below as needed
class Config:
    #Models to fine-tune
    MODEL_NAMES = {
        # "RoBERTa-Large": "FacebookAI/roberta-large"
        #Example:  
        "BERT-Base": "bert-base-uncased",
        "DistilBERT": "distilbert/distilbert-base-uncased",    
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "E5-Base": "intfloat/e5-base",
        "DeBERTa-Base": "microsoft/deberta-base"
    }
    BASE_PATH = "data/processed_data/md_offense" #Path to the processed data
    OUTPUT_DIR = "fine_tuned_models" # where to save fine-tuned models and logs

    #Training configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    SEEDS = [3] #Add more seeds here if needed [1,2,3]
    SAVE_TOP_K = 5 #Keep top k model checkpoints 

# Load Dataset
def load_dataset_for_training():
    base_path = Config.BASE_PATH
    split_datasets = {"train": [], "dev": [], "test": []}

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path {base_path} does not exist!")

    for split in ["train", "dev", "test"]:
        split_path = os.path.join(base_path, split)
        if os.path.isdir(split_path):
            for ambiguity_type in os.listdir(split_path):
                ambiguity_path = os.path.join(split_path, ambiguity_type)
                if os.path.isdir(ambiguity_path):
                    try:
                        dataset = load_from_disk(ambiguity_path)
                        split_datasets[split].append(dataset)
                    except Exception as e:
                        print(f"Error loading {split}/{ambiguity_type}: {e}")

    combined_dataset = {split: concatenate_datasets(split_datasets[split]) for split in split_datasets if split_datasets[split]}
    return DatasetDict(combined_dataset)

#Preprocess Data
def preprocess_data(dataset, tokenizer):
    def tokenize_function(example):
        return tokenizer(example["Text"], truncation=True, padding="max_length", max_length=512)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(lambda x: {"labels": int(x["Offensive_binary_label"])}, batched=False)
    return dataset

#Compute Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": accuracy, "f1_macro": f1}

#Logging Callback
class ContinuousLoggingCallback(TrainerCallback):
    def __init__(self, model_name, seed):
        log_dir = f"{Config.OUTPUT_DIR}/{model_name}/seed_{seed}/logs"  #Logs inside seed_X/logs
        os.makedirs(log_dir, exist_ok=True)

        self.log_paths = {
            "train": f"{log_dir}/trainer_logs.csv",
            "eval": f"{log_dir}/eval_results.csv"
        }
        self.headers_written = {key: not os.path.exists(path) for key, path in self.log_paths.items()}

    def log_to_csv(self, file_path, data, header_flag):
        with open(file_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if header_flag:
                writer.writeheader()
                header_flag = False  #Write headers only once
            writer.writerow(data)
        return header_flag

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" not in logs:
            logs["epoch"] = state.epoch
            logs["global_step"] = state.global_step
            self.headers_written["train"] = self.log_to_csv(self.log_paths["train"], logs, self.headers_written["train"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            metrics["epoch"] = state.epoch
            metrics["global_step"] = state.global_step
            self.headers_written["eval"] = self.log_to_csv(self.log_paths["eval"], metrics, self.headers_written["eval"])

def train_model(model_name, model_path):
    dataset_dict = load_dataset_for_training()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #Preprocess dataset
    dataset = preprocess_data(dataset_dict, tokenizer)

    for seed in Config.SEEDS:
        #Set reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #Split dataset
        train_dataset, dev_dataset, test_dataset = dataset["train"], dataset["dev"], dataset["test"]
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(Config.DEVICE)

        #Create  directory structure
        seed_output_dir = f"{Config.OUTPUT_DIR}/{model_name}/seed_{seed}"
        log_dir = f"{seed_output_dir}/logs"  
        tensorboard_dir = f"{seed_output_dir}/tensorboard"
        os.makedirs(seed_output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=seed_output_dir,  # Model checkpoints inside seed_X/
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            logging_dir=tensorboard_dir,  # TensorBoard logs go here
            logging_steps=100,
            save_total_limit=Config.SAVE_TOP_K,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to="tensorboard",
            seed=seed,
        )

        #Callbacks
        early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
        logging_callback = ContinuousLoggingCallback(model_name, seed)

        #Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping, logging_callback],  #Use the modified logging callback
        )

        #Train
        trainer.train()

        #Save evaluation results
        eval_results = trainer.evaluate(dev_dataset)
        test_results = trainer.evaluate(test_dataset)

        with open(f"{log_dir}/dev_eval_results.json", "w") as result_file:
            json.dump(eval_results, result_file, indent=2)

        with open(f"{log_dir}/test_eval_results.json", "w") as result_file:
            json.dump(test_results, result_file, indent=2)

        # Save trained model
        model.save_pretrained(seed_output_dir)
        tokenizer.save_pretrained(seed_output_dir)

        print(f"Model {model_name} trained on seed {seed} and logs are stored inside {log_dir}")


# Run Training
def main():
    for model_name, model_path in Config.MODEL_NAMES.items():
        train_model(model_name, model_path)

if __name__ == "__main__":
    main()
