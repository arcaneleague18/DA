import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

def pearson_corr(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    return numerator / denominator

cols = df.columns

print("\nPairwise Correlations (Manual):")

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r = pearson_corr(df[cols[i]], df[cols[j]])
        print(f"{cols[i]} vs {cols[j]} => r = {r:.4f}")

corr_matrix = pd.DataFrame(index=cols, columns=cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        corr_matrix.iloc[i, j] = pearson_corr(df[cols[i]], df[cols[j]])

print("\nCorrelation Matrix:\n", corr_matrix.astype(float).round(4))

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix.astype(float),
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
)

plt.title("Correlation Heatmap - Iris Dataset (Manual)")
plt.tight_layout()
plt.show()
