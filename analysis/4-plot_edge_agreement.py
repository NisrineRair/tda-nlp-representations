import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === CONFIG ===
EDGE_FILE = "analysis/results_excel/edge_agreement_summary.xlsx"
OUTPUT_DIR = "analysis/figures"
PLOT_BY = "lens"  # Options: "ambiguity" or "lens"
SHOW_ERROR_BARS = False  # Set to False to disable SD bars

# === Load Data ===
df = pd.read_excel(EDGE_FILE)

# === Preprocessing ===
df["ambiguity"] = pd.Categorical(df["ambiguity"], categories=["A0", "A+", "A++"], ordered=True)

# === Plot Setup ===
sns.set_style("whitegrid")
palette = ["#1f77b4", "#ff7f0e"]  # Base: blue, Fine-tuned: orange

if PLOT_BY == "ambiguity":
    x_col = "ambiguity"
    title = "Label-Aligned Edge Consistency by Ambiguity"
    output_file = os.path.join(OUTPUT_DIR, "edge_agreement_by_ambiguity.pdf")
elif PLOT_BY == "lens":
    x_col = "lens"
    title = "Label-Aligned Edge Consistency by Lens"
    output_file = os.path.join(OUTPUT_DIR, "edge_agreement_by_lens.pdf")
    df[x_col] = df[x_col].astype(str)  # Ensure string for plotting
else:
    raise ValueError("PLOT_BY must be either 'ambiguity' or 'lens'")

# === Plot ===
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df,
    x=x_col,
    y="edge_agreement_unweighted",
    hue="version",
    errorbar="sd" if SHOW_ERROR_BARS else None,
    capsize=0.15,
    palette=palette
)

plt.ylim(0, 1)
plt.xlabel(x_col.capitalize())
plt.ylabel("Edge Agreement (Unweighted)")
plt.title(title)
plt.legend(title="Model Version")
if PLOT_BY == "lens":
    plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# === Save ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(output_file, dpi=300)
print(f"Plot saved to: {output_file}")
