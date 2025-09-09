import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIG ===
COMPONENT_FILE = "analysis/results_excel/component_purity_summary.xlsx"
FIGURE_DIR = "analysis/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

VIOLIN_AGG_FIG = os.path.join(FIGURE_DIR, "violin_component_purity_ordered.pdf")
VIOLIN_LENS_FIG = os.path.join(FIGURE_DIR, "violin_component_purity_all_lenses.pdf")

# === Load data ===
df = pd.read_excel(COMPONENT_FILE)

# === Order ambiguity levels ===
ordered_levels = ["A0", "A+", "A++"]
df["ambiguity"] = pd.Categorical(df["ambiguity"], categories=ordered_levels, ordered=True)

# === Set style and palette ===
sns.set_style("whitegrid")
palette = ["#1f77b4", "#ff7f0e"]  # Base, Fine-tuned

# === Plot 1: Aggregated violin plot ===
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.violinplot(
    data=df,
    x="ambiguity",
    y="weighted_node_purity",
    hue="version",
    split=True,
    inner="quartile",
    linewidth=1.2,
    density_norm="width",
    bw_method=0.3,
    palette=palette,
    ax=ax1
)
ax1.set_xlabel("Ambiguity Level")
ax1.set_ylabel("Component Purity")
ax1.set_ylim(0, 1)
ax1.legend(title="Model Version")
fig1.tight_layout()
fig1.savefig(VIOLIN_AGG_FIG, dpi=300)
plt.close(fig1)
print(f"Saved aggregated violin plot: {VIOLIN_AGG_FIG}")

# === Plot 2: Per-lens subplot violin plots ===
lenses = df["lens"].unique()
n_lenses = len(lenses)
cols = 3
rows = math.ceil(n_lenses / cols)

fig2, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharey=True)
axes = axes.flatten()

for i, lens_name in enumerate(lenses):
    ax = axes[i]
    subset = df[df["lens"] == lens_name]
    sns.violinplot(
        data=subset,
        x="ambiguity",
        y="weighted_node_purity",
        hue="version",
        split=True,
        inner="quartile",
        linewidth=1.2,
        density_norm="width",
        bw_method=0.3,
        palette=palette,
        ax=ax
    )
    ax.set_title(lens_name, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Purity")
    ax.set_ylim(0, 1)
    if i == 0:
        ax.legend(title="Model Version")
    else:
        ax.get_legend().remove()

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig2.delaxes(axes[j])

fig2.suptitle("Component Purity per Lens", fontsize=14)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(VIOLIN_LENS_FIG, dpi=300)
plt.close(fig2)
print(f"Saved per-lens violin plots: {VIOLIN_LENS_FIG}")
