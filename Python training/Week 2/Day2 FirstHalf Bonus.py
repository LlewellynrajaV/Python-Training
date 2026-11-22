import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

start_rf = time.perf_counter()
rf = RandomForestClassifier(
    n_estimators=200, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_time = time.perf_counter() - start_rf
rf_acc = accuracy_score(y_test, rf.predict(X_test))

start_et = time.perf_counter()
et_default = ExtraTreesClassifier(
    n_estimators=200, bootstrap=False, random_state=42, n_jobs=-1
)
et_default.fit(X_train, y_train)
et_default_time = time.perf_counter() - start_et
et_default_acc = accuracy_score(y_test, et_default.predict(X_test))

et_oob = ExtraTreesClassifier(
    n_estimators=200, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
)
et_oob.fit(X_train, y_train)
et_oob_acc = accuracy_score(y_test, et_oob.predict(X_test))

comparison_df = pd.DataFrame({
    "Model": [
        "RandomForest (bootstrap=True)",
        "ExtraTrees (bootstrap=False, default)",
        "ExtraTrees (bootstrap=True, OOB)",
    ],
    "Test Accuracy": [rf_acc, et_default_acc, et_oob_acc],
    "OOB Score": [rf.oob_score_, None, et_oob.oob_score_],
    "Fit Time (s)": [round(rf_time, 4), round(et_default_time, 4), None],
})

print("RandomForest vs ExtraTrees Comparison:")
print(comparison_df.to_string(index=False))

imp_df = pd.DataFrame(
    {
        "Feature": wine.feature_names,
        "RandomForest": rf.feature_importances_,
        "ExtraTrees": et_default.feature_importances_,
    }
).sort_values(by="RandomForest", ascending=True)

y_pos = range(len(imp_df))
height = 0.38

plt.figure(figsize=(9, 6))
plt.barh(
    [p - height / 2 for p in y_pos],
    imp_df["RandomForest"],
    height=height,
    label="RandomForest",
    color="teal",
)
plt.barh(
    [p + height / 2 for p in y_pos],
    imp_df["ExtraTrees"],
    height=height,
    label="ExtraTrees",
    color="coral",
)
plt.yticks(y_pos, imp_df["Feature"])
plt.xlabel("Feature Importance")
plt.title("Feature Importances: RandomForest vs. ExtraTrees")
plt.legend()
plt.tight_layout()
plt.savefig("bonus_rf_vs_extratrees.png", dpi=300)
plt.show()