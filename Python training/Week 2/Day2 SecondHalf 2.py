import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X_blobs, _ = make_blobs(
    n_samples=300, centers=4, cluster_std=0.60, random_state=7
)

k_values = range(2, 9)
inertias = []
sil_scores = []

for k in k_values:
  km = KMeans(n_clusters=k, n_init=10, random_state=42)
  labels = km.fit_predict(X_blobs)
  inertias.append(km.inertia_)
  score = silhouette_score(X_blobs, labels)
  sil_scores.append(score)
  print(f"k={k} | Inertia={km.inertia_:8.2f} | Silhouette Score={score:.4f}")

best_k = k_values[int(np.argmax(sil_scores))]
print(
    f"\nBest k by silhouette score: {best_k} (score ="
    f" {max(sil_scores):.4f})"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(list(k_values), inertias, marker="o")
axes[0].set_title("Elbow Method (Inertia vs. k)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)

axes[1].plot(list(k_values), sil_scores, marker="s", color="tab:orange")
axes[1].set_title("Silhouette Score vs. k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)

plt.tight_layout()
plt.show()

km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
final_labels = km_final.fit_predict(X_blobs)
centroids = km_final.cluster_centers_

plt.figure(figsize=(7, 5.5))
plt.scatter(
    X_blobs[:, 0],
    X_blobs[:, 1],
    c=final_labels,
    cmap="viridis",
    s=35,
    alpha=0.8,
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    label="Centroids",
)
plt.title(f"K-Means Clusters & Centroids (k={best_k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

rng = np.random.default_rng(42)
segment_centers = [
    [25, 20],  # Low income, low spending
    [25, 80],  # Low income, high spending
    [55, 50],  # Average income, average spending
    [85, 18],  # High income, low spending
    [85, 82],  # High income, high spending
]
mall_chunks = [
    rng.normal(loc=c, scale=[7, 7], size=(40, 2)) for c in segment_centers
]
mall_df = pd.DataFrame(
    np.vstack(mall_chunks), columns=["Annual_Income", "Spending_Score"]
).clip(1, 100)

km_mall = KMeans(n_clusters=5, n_init=10, random_state=42)
mall_df["Segment"] = km_mall.fit_predict(mall_df)

plt.figure(figsize=(7.5, 5.5))
plt.scatter(
    mall_df["Annual_Income"],
    mall_df["Spending_Score"],
    c=mall_df["Segment"],
    cmap="tab10",
    s=40,
    alpha=0.85,
)
plt.scatter(
    km_mall.cluster_centers_[:, 0],
    km_mall.cluster_centers_[:, 1],
    c="black",
    marker="P",
    s=180,
    label="Centroids",
)
plt.title("Mall Customers Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()

summary = mall_df.groupby("Segment")[["Annual_Income", "Spending_Score"]].mean()
print("\nSegment Means:")
print(summary.round(1))