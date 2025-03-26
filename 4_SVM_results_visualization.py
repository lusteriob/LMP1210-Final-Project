import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# Load data
from pathlib import Path
df = pd.read_csv(Path("models") / "svm_nested_cv_multiboss" / "metrics_summary.csv")


# === Summary Statistics ===
# Only include numeric columns for aggregation
numeric_cols = ["balanced_accuracy", "f1", "precision", "recall", "roc_auc"]
summary = df.groupby("feature_set")[numeric_cols].agg(['mean', 'std'])
print("=== Summary Statistics ===")
print(summary[["balanced_accuracy", "f1", "precision", "recall", "roc_auc"]])
print("\n")

# === Paired t-tests ===
def paired_ttests(metric):
    print(f"\n=== Paired t-tests for {metric} ===")
    sets = df["feature_set"].unique()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            group1 = df[df["feature_set"] == sets[i]][metric].values
            group2 = df[df["feature_set"] == sets[j]][metric].values
            t_stat, p_val = ttest_rel(group1, group2)
            print(f"{sets[i]} vs {sets[j]}: t={t_stat:.4f}, p={p_val:.4f}")

for metric in ["balanced_accuracy", "f1", "precision", "recall", "roc_auc"]:
    paired_ttests(metric)

# === Bar plots with error bars ===
def plot_metric_bar(metric):
    stats = df.groupby("feature_set")[metric].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar(stats["feature_set"], stats["mean"], yerr=stats["std"], capsize=6)
    plt.title(f"{metric.replace('_', ' ').title()} by Feature Set")
    plt.ylabel(metric.replace("_", " ").title())
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

for m in ["balanced_accuracy", "f1", "precision", "recall", "roc_auc"]:
    plot_metric_bar(m)

# === Heatmaps of fold-wise performance ===
def plot_fold_heatmap(metric):
    pivot = df.pivot(index="fold", columns="feature_set", values=metric).sort_index()
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": metric.replace('_', ' ').title()})
    plt.title(f"{metric.replace('_', ' ').title()} per Fold and Feature Set")
    plt.xlabel("Feature Set")
    plt.ylabel("Fold")
    plt.tight_layout()
    plt.show()

for m in ["balanced_accuracy", "f1", "precision", "recall", "roc_auc"]:
    plot_fold_heatmap(m)
