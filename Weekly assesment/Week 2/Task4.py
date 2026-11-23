import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("loan_data.csv")
X = df.drop(columns=["default"])
y = df["default"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)


def add_features(data_in):
    out = data_in.copy()
    dt = pd.to_datetime(out["application_date"])
    out["app_month"] = dt.dt.month
    out["app_dayofweek"] = dt.dt.dayofweek
    out["income_to_loan"] = out["annual_income"] / (out["loan_amount"] + 1)
    out["log_income"] = np.log1p(out["annual_income"].clip(lower=0))
    return out


num_cols = [
    "age", "annual_income", "credit_score", "debt_to_income", "employment_years",
    "loan_amount", "app_month", "app_dayofweek", "income_to_loan", "log_income",
]
ord_cols = ["education"]
cat_cols = ["employment_type", "home_ownership"]

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ord", OrdinalEncoder(categories=[["High School", "Bachelor", "Master", "PhD"]],
                                             handle_unknown="use_encoded_value", unknown_value=-1))]), ord_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])

models = {
    "LogisticRegression": Pipeline([("fe", FunctionTransformer(add_features, validate=False)), ("prep", prep),
                                    ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
    "DecisionTree": Pipeline([("fe", FunctionTransformer(add_features, validate=False)), ("prep", prep),
                              ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))]),
    "RandomForest": Pipeline([("fe", FunctionTransformer(add_features, validate=False)), ("prep", prep),
                              ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
    "GradientBoosting": Pipeline([("fe", FunctionTransformer(add_features, validate=False)), ("prep", prep),
                                  ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                         max_depth=4, random_state=42))]),
}

print("--- Classifier Metrics ---")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    print(f"{name:18s} | Acc={accuracy_score(y_val, preds):.4f} Prec={precision_score(y_val, preds, zero_division=0):.4f} "
          f"Rec={recall_score(y_val, preds, zero_division=0):.4f} F1={f1_score(y_val, preds, zero_division=0):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))

plt.figure(figsize=(6.5, 4.5))
for name in ["LogisticRegression", "RandomForest", "GradientBoosting"]:
    probs = models[name].predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_val, probs):.2f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("task4_roc.png")
plt.show()
plt.close()

probs_gb = models["GradientBoosting"].predict_proba(X_val)[:, 1]
precs, recs, ths = precision_recall_curve(y_val, probs_gb)
valid = np.where(recs[:-1] >= 0.75)[0]
best_i = valid[np.argmax(precs[:-1][valid])]
thresh = ths[best_i]
print(f"\nChosen threshold: {thresh:.4f} -> Recall: {recs[best_i]:.2f}, Precision: {precs[best_i]:.2f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_rf = GridSearchCV(
    Pipeline([("fe", FunctionTransformer(add_features, validate=False)), ("prep", prep),
              ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))]),
    {"clf__n_estimators": [100, 200], "clf__max_depth": [4, 6, None], "clf__min_samples_split": [2, 10]},
    cv=cv, scoring="roc_auc", n_jobs=-1,
).fit(X_train, y_train)
print("\nGridSearchCV RF Best:", gs_rf.best_params_, "| Score:", round(gs_rf.best_score_, 4))

rs_gb = RandomizedSearchCV(
    Pipeline([("fe", FunctionTransformer(add_features, validate=False)), ("prep", prep),
              ("clf", HistGradientBoostingClassifier(random_state=42))]),
    {"clf__learning_rate": [0.01, 0.03, 0.05, 0.1], "clf__max_depth": [3, 4, 6, None],
     "clf__min_samples_leaf": [10, 20, 40], "clf__l2_regularization": [0.0, 0.1, 1.0, 5.0]},
    n_iter=15, cv=cv, scoring="roc_auc", random_state=42, n_jobs=-1,
).fit(X_train, y_train)
print("RandomizedSearchCV GB Best:", rs_gb.best_params_, "| Score:", round(rs_gb.best_score_, 4))

X_reg = df.drop(columns=["default", "loan_amount"])
y_reg = df["loan_amount"]
X_r_tr, y_r_tr = X_reg.loc[X_train.index], y_reg.loc[X_train.index]
X_r_va, y_r_va = X_reg.loc[X_val.index], y_reg.loc[X_val.index]


def add_reg_features(data_in):
    out = data_in.copy()
    dt = pd.to_datetime(out["application_date"])
    out["app_month"] = dt.dt.month
    out["app_dayofweek"] = dt.dt.dayofweek
    out["log_income"] = np.log1p(out["annual_income"].clip(lower=0))
    out["inc_per_yr"] = out["annual_income"] / (out["employment_years"] + 1)
    return out


r_num = ["age", "annual_income", "credit_score", "debt_to_income", "employment_years", "app_month", "app_dayofweek", "log_income", "inc_per_yr"]
r_prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), r_num),
    ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ord", OrdinalEncoder(categories=[["High School", "Bachelor", "Master", "PhD"]],
                                             handle_unknown="use_encoded_value", unknown_value=-1))]), ord_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])

reg_pipe = Pipeline([("fe", FunctionTransformer(add_reg_features, validate=False)), ("prep", r_prep), ("reg", Ridge(alpha=10.0))]).fit(X_r_tr, y_r_tr)
reg_pred = reg_pipe.predict(X_r_va)
base_pred = np.full_like(y_r_va, y_r_tr.mean(), dtype=float)

print(f"\nRegression -> MAE: {mean_absolute_error(y_r_va, reg_pred):,.2f} | RMSE: {np.sqrt(mean_squared_error(y_r_va, reg_pred)):,.2f} | R2: {r2_score(y_r_va, reg_pred):.4f}")
print(f"Baseline   -> MAE: {mean_absolute_error(y_r_va, base_pred):,.2f} | RMSE: {np.sqrt(mean_squared_error(y_r_va, base_pred)):,.2f} | R2: {r2_score(y_r_va, base_pred):.4f}")

best_model = rs_gb if rs_gb.best_score_ >= gs_rf.best_score_ else gs_rf
test_probs = best_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= thresh).astype(int)
print(f"\nBest model ROC-AUC: {roc_auc_score(y_test, test_probs):.2f} | F1: {f1_score(y_test, test_preds):.2f}")