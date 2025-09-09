import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIG ===
LABEL_FILE = "analysis/results_excel/component_level_summary_test_offensive_majority.xlsx"
PRED_FILE = "analysis/results_excel/component_level_summary_test_model_prediction.xlsx"
OUTPUT_PLOT = "analysis/figures/component_purity_violinplot_all_distributions.pdf"

# === Load files ===
df_label = pd.read_excel(LABEL_FILE)
df_pred = pd.read_excel(PRED_FILE)

# === Keep fine-tuned only
df_label = df_label[df_label["version"] == "fine_tuned"]
df_pred = df_pred[df_pred["version"] == "fine_tuned"]

# === Rename and tag sources
df_label = df_label.rename(columns={"weighted_node_purity": "purity"})
df_label["source"] = "Ground Truth"
df_pred = df_pred.rename(columns={"weighted_node_purity": "purity"})
df_pred["source"] = "Prediction"

# === Concatenate both
merge_cols = ["model", "version", "subset", "ambiguity", "lens", "param", "component_id", "purity", "source"]
df_combined = pd.concat([df_label[merge_cols[:-2] + ["purity", "source"]], 
                         df_pred[merge_cols[:-2] + ["purity", "source"]]])

# === Order ambiguity levels
df_combined["ambiguity"] = pd.Categorical(df_combined["ambiguity"], 
                                          categories=["A0", "A+", "A++"], ordered=True)

# === Plot settings
sns.set_style("whitegrid")
sns.set_palette("Set2")

plt.figure(figsize=(10, 6))
sns.violinplot(
    data=df_combined,
    x="ambiguity",
    y="purity",
    hue="source",
    split=True,
    inner="quartile",
    bw=0.2
)

plt.ylim(0, 1)
plt.xlabel("Ambiguity Level")
plt.ylabel("Component Purity")
plt.title("Component Purity Distribution (Ground Truth vs Prediction)")
plt.legend(title="Source", loc="upper left")
plt.tight_layout()

# === Save
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

print(f"Violin plot saved to: {OUTPUT_PLOT}")
