import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# === Step 1: Load data ===
deep_prot_df = pd.read_csv("input_sequences/deep_proteomics.csv")
model_output_df = pd.read_csv("test_output_transformer_classifier_w-20_emb-2560.csv")

# === Step 2: Merge on uniprotID ===
merged_df = pd.merge(model_output_df, deep_prot_df, on="uniprotID", how="inner")

# === Step 3: Rename columns for convenience ===
merged_df.rename(columns={
    "score": "Model_Score",
    "Enrichment from input(Experimental data)": "Experimental_Enrichment"
}, inplace=True)

# === Step 4: Drop missing values (if any) ===
merged_df = merged_df.dropna(subset=["Model_Score", "Experimental_Enrichment"])

# === Step 5: Scatter plot: model score vs experimental enrichment ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=merged_df,
    x="Model_Score",
    y="Experimental_Enrichment",
    alpha=0.6,
    edgecolor=None
)
plt.xlabel("Model Score (Predicted Binding Confidence)")
plt.ylabel("Experimental Enrichment Level")
plt.title("Model Score vs Experimental Binding Strength")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 6: Correlation analysis ===
pearson_corr, _ = pearsonr(merged_df["Model_Score"], merged_df["Experimental_Enrichment"])
spearman_corr, _ = spearmanr(merged_df["Model_Score"], merged_df["Experimental_Enrichment"])

print(f"Pearson correlation: {pearson_corr:.3f}")
print(f"Spearman correlation: {spearman_corr:.3f}")
