import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


dt_gini = DecisionTreeClassifier(random_state=42, criterion="gini")
dt_gini.fit(X_train, y_train)
gini_acc = accuracy_score(y_test, dt_gini.predict(X_test))

dt_entropy = DecisionTreeClassifier(random_state=42, criterion="entropy")
dt_entropy.fit(X_train, y_train)
entropy_acc = accuracy_score(y_test, dt_entropy.predict(X_test))

print(f"Decision Tree (Gini) Test Accuracy:    {gini_acc:.4f}")
print(f"Decision Tree (Entropy) Test Accuracy: {entropy_acc:.4f}")

param_grid = {
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": [1, 2, 3, 5, 10],
}
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_
best_dt_acc = accuracy_score(y_test, best_dt.predict(X_test))
print("Best DT Params:", grid_search.best_params_)
print(f"Tuned Decision Tree Test Accuracy: {best_dt_acc:.4f}")

plt.figure(figsize=(16, 8))
plot_tree(
    DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train),
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree (max_depth=3) on Wine Dataset")
plt.tight_layout()
plt.savefig("decision_tree_depth3.png", dpi=300)
plt.show()

dt_importances = (
    pd.DataFrame(
        {"Feature": wine.feature_names, "Importance": best_dt.feature_importances_}
    )
    .sort_values(by="Importance", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 5 Most Important Features (Decision Tree):")
print(dt_importances.head(5).to_string(index=False))

