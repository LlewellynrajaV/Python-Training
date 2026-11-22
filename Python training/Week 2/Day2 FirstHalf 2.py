import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_tree_acc = accuracy_score(y_test, single_tree.predict(X_test))

rf = RandomForestClassifier(
    n_estimators=200, oob_score=True, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(
    f"Single tree test accuracy: {single_tree_acc:.2f} | Random forest:"
    f" {rf_acc:.2f} | OOB Score: {rf.oob_score_:.2f}"
)

n_estimators_list = [10, 50, 100, 300]
accuracies = []

for n in n_estimators_list:
  model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
  model.fit(X_train, y_train)
  accuracies.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(7, 4))
plt.plot(n_estimators_list, accuracies, marker="o", linewidth=2, color="navy")
plt.title("Random Forest Test Accuracy vs. n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Test Accuracy")
plt.xticks(n_estimators_list)
plt.ylim(0.85, 1.02)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("rf_n_estimators_accuracy.png", dpi=300)
plt.show()

rf_importances = pd.DataFrame(
    {"Feature": wine.feature_names, "Importance": rf.feature_importances_}
).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(rf_importances["Feature"], rf_importances["Importance"], color="teal")
plt.title("Random Forest Feature Importances (Wine Dataset)")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("rf_feature_importances.png", dpi=300)
plt.show()

