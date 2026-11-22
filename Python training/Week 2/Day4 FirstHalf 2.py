import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X, y = make_classification(
    n_samples=2000, weights=[0.95, 0.05], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression().fit(X_train, y_train)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
xgb = XGBClassifier(eval_metric="logloss", random_state=42).fit(
    X_train, y_train
)

models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}

plt.figure()
rows = []

for name, model in models.items():
  prob = model.predict_proba(X_test)[:, 1]
  pred = model.predict(X_test)
  fpr, tpr, _ = roc_curve(y_test, prob)
  auc = roc_auc_score(y_test, prob)
  f1 = f1_score(y_test, pred)
  print(name, "AUC:", round(auc, 4))
  plt.plot(fpr, tpr, label=f"{name} ({auc:.2f})")
  rows.append({"Model": name, "AUC": round(auc, 4), "F1": round(f1, 4)})

plt.plot([0, 1], [0, 1], "--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("\nTable:")
print(pd.DataFrame(rows))

rf_prob = rf.predict_proba(X_test)[:, 1]
prec, rec, thresh = precision_recall_curve(y_test, rf_prob)

for p, r, t in zip(prec[:-1], rec[:-1], thresh):
  if r >= 0.80:
    print("\nThreshold:", t)
    print("Recall:", r)
    print("Precision:", p)
    break

plt.figure()
for name, model in models.items():
  prob = model.predict_proba(X_test)[:, 1]
  p, r, _ = precision_recall_curve(y_test, prob)
  plt.plot(r, p, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()