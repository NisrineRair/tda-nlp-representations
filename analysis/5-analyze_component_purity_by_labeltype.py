import pandas as pd

# === CONFIG ===
LABEL_FILE = "analysis/results_excel/component_level_summary_test_offensive_majority.xlsx"
PRED_FILE = "analysis/results_excel/component_level_summary_test_model_prediction.xlsx"
PURITY_THRESHOLD = 0.9
OUTPUT_FILE = "analysis/results_excel/prediction_vs_label_purity_summary.xlsx"

# === Load files ===
df_label = pd.read_excel(LABEL_FILE)
df_pred = pd.read_excel(PRED_FILE)

# === Filter fine-tuned only ===
df_label = df_label[df_label["version"] == "fine_tuned"]
df_pred = df_pred[df_pred["version"] == "fine_tuned"]

# === Rename columns ===
df_label = df_label.rename(columns={
    "weighted_node_purity": "label_purity",
    "majority_label": "majority_label_true"
})
df_pred = df_pred.rename(columns={
    "weighted_node_purity": "prediction_purity",
    "majority_label": "majority_label_pred"
})

# === Add flags for ≥90% purity
df_label["label_is_pure"] = df_label["label_purity"] >= PURITY_THRESHOLD
df_pred["prediction_is_pure"] = df_pred["prediction_purity"] >= PURITY_THRESHOLD

# === Merge on common component keys
merge_cols = ["model", "version", "subset", "ambiguity", "lens", "param", "component_id"]
df = pd.merge(
    df_label[merge_cols + ["label_purity", "label_is_pure", "majority_label_true"]],
    df_pred[merge_cols + ["prediction_purity", "prediction_is_pure", "majority_label_pred"]],
    on=merge_cols
)

# === Compute majority match (true vs predicted nodewise label)
df["majority_match"] = df["majority_label_true"] == df["majority_label_pred"]

# === Filter to components where both label and prediction purity ≥ 90%
df_filtered = df[df["label_is_pure"] & df["prediction_is_pure"]]
match_90 = (
    df_filtered.groupby(["ambiguity", "lens"])
    .agg(majority_match_90=("majority_match", "sum"),
         total_90=("component_id", "count"))
    .reset_index()
)

# === Group by ambiguity × lens and aggregate
summary = (
    df.groupby(["ambiguity", "lens"])
      .agg(
          total_components=("component_id", "count"),
          pure_label=("label_is_pure", "sum"),
          pure_pred=("prediction_is_pure", "sum"),
          majority_match=("majority_match", "sum")
      )
      .reset_index()
)

# === Merge with filtered match counts
summary = pd.merge(summary, match_90, on=["ambiguity", "lens"], how="left")

# === Compute percentages
summary["% ≥90% Pure (Label)"] = 100 * summary["pure_label"] / summary["total_components"]
summary["% ≥90% Pure (Prediction)"] = 100 * summary["pure_pred"] / summary["total_components"]
summary["% Majority Match"] = 100 * summary["majority_match"] / summary["total_components"]
summary["% Majority Match (≥90% Pure)"] = 100 * summary["majority_match_90"] / summary["total_90"]
summary["Delta (Pred − Label)"] = summary["% ≥90% Pure (Prediction)"] - summary["% ≥90% Pure (Label)"]

# === Final table per lens
per_lens = summary[["ambiguity", "lens",
                    "% ≥90% Pure (Label)", "% ≥90% Pure (Prediction)",
                    "% Majority Match", "% Majority Match (≥90% Pure)",
                    "Delta (Pred − Label)"]].round(1)

# === Aggregated table across lenses (per ambiguity)
agg = (
    per_lens.groupby("ambiguity")
    .agg({
        "% ≥90% Pure (Label)": ['mean', 'std'],
        "% ≥90% Pure (Prediction)": ['mean', 'std'],
        "% Majority Match": ['mean', 'std'],
        "% Majority Match (≥90% Pure)": ['mean', 'std'],
        "Delta (Pred − Label)": ['mean', 'std']
    })
)

# Flatten column names
agg.columns = [' '.join(col).strip() for col in agg.columns.values]
agg = agg.reset_index()
agg = agg.round(1)

# === Save both sheets
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    per_lens.to_excel(writer, sheet_name="Per Lens Summary", index=False)
    agg.to_excel(writer, sheet_name="Aggregated by Ambiguity", index=False)

print("Saved full summary with lens and aggregate stats to:", OUTPUT_FILE)
