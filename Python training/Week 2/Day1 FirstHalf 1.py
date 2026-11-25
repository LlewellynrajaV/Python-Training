import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"F1 (macro): {f1_macro:.2f}")
print("Confusion matrix:\n", cm)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

cluster_sizes = np.bincount(clusters)
print("KMeans cluster sizes:", cluster_sizes.tolist())

comparison_df = pd.crosstab(
    index=pd.Series(clusters, name="Cluster"),
    columns=pd.Series([iris.target_names[i] for i in y], name="Species")
)
print("\nCluster vs. True Species:\n", comparison_df)