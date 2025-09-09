import os
import pandas as pd
from datasets import Dataset, DatasetDict

def load_md_offense():
    """Load MD-Offense dataset from TSV and return as a Hugging Face Dataset."""
    file_path = "data/raw_data/MD-Agreement_dataset.tsv"

    required_cols = ["ID", "Text", "Agreement_level", "Offensive_binary_label", 
                     "Individual_Annotations", "Annotators_ID", "Domain"]
    
    df = pd.read_csv(file_path, sep="\t", quoting=3, dtype=str, usecols=required_cols)
    df["Text"] = df["Text"].astype(str).str.strip()
    
    return Dataset.from_pandas(df)

def split_by_dataset_type(md_dataset):
    """Split MD-Offense dataset into train, test, and validation based on ID prefix."""
    return DatasetDict({
        split: md_dataset.filter(lambda x: f"_{split}" in x["ID"])
        for split in ["train", "test", "dev"]
    })

def split_by_agreement_level(dataset):
    """Further split dataset into A0, A+, and A++ subsets."""
    return {level: dataset.filter(lambda x: x["Agreement_level"] == level)
            for level in ["A0", "A+", "A++"]}

def compute_statistics(dataset, split_name, subset_name):
    """Compute basic statistics of the dataset and return as a dictionary."""
    total_samples = len(dataset)
    offensive_count = sum(1 for x in dataset["Offensive_binary_label"] if x == "1")
    
    return {
        "Split": split_name,
        "Agreement_Level": subset_name,
        "Total Samples": total_samples,
        "Offensive Samples": offensive_count,
        "Non-Offensive Samples": total_samples - offensive_count,
        "Offensive Ratio": offensive_count / total_samples if total_samples else 0
    }

def main():
    """Main function to load, process, split, and save MD-Offense structured dataset."""
    print("Loading MD-Offense dataset...")
    md_offense_dataset = load_md_offense()

    print("Splitting into Train, Test, and Validation sets...")
    dataset_splits = split_by_dataset_type(md_offense_dataset)

    # Collect statistics for all subsets in a single list 
    all_statistics = []

    print("Processing each set and saving data...")
    for split_name, split_data in dataset_splits.items():
        subset_splits = split_by_agreement_level(split_data)
        
        for subset_name, subset_data in subset_splits.items():
            output_path = os.path.join("data", "processed_data", "md_offense", split_name, subset_name)
            os.makedirs(output_path, exist_ok=True)

            # Save dataset
            subset_data.save_to_disk(output_path)

            # Compute statistics and collect them
            stats = compute_statistics(subset_data, split_name, subset_name)
            all_statistics.append(stats)

            print(f" Saved: {split_name}/{subset_name} ({len(subset_data)} samples)")

    # Save all statistics in a single CSV file
    stats_df = pd.DataFrame(all_statistics)
    stats_output_path = os.path.join("data", "processed_data", "md_offense", "statistics.csv")
    stats_df.to_csv(stats_output_path, index=False)
    print(f"Unified statistics saved: {stats_output_path}")

    print("All subsets processed and saved successfully!")

if __name__ == "__main__":
    main()
