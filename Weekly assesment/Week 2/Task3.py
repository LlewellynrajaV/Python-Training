import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
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
                              ("clf", RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42, n_jobs=-1))]),
    "GradientBoosting (XGB proxy)": Pipeline([("fe", FunctionTransformer(add_features, validate=False)), ("prep", prep),
                                              ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                                     max_depth=4, early_stopping=True,
                                                                                     validation_fraction=0.15, random_state=42))]),
}

print("--- Validation Scores ---")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_val, pipe.predict(X_val))
    auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    print(f"{name:28s} | Val Acc: {acc:.4f} | Val ROC-AUC: {auc:.4f}")

print("RandomForest OOB Score:", round(models["RandomForest"].named_steps["clf"].oob_score_, 4))

probs_5 = models["LogisticRegression"].predict_proba(X_val.iloc[:5])
preds_5 = models["LogisticRegression"].predict(X_val.iloc[:5])
print("\n--- 5 Rows: predict_proba vs predict ---")
for i in range(5):
    print(f"Row {i+1}: P(0)={probs_5[i, 0]:.3f}, P(1)={probs_5[i, 1]:.3f} -> predict={preds_5[i]}")

ohe_cols = list(models["RandomForest"].named_steps["prep"].named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols))
all_cols = num_cols + ord_cols + ohe_cols

rf_imp = pd.Series(models["RandomForest"].named_steps["clf"].feature_importances_, index=all_cols).sort_values(ascending=False)
print("\nTop 5 RandomForest Features:\n", rf_imp.head(5).round(4))

X_val_t = models["GradientBoosting (XGB proxy)"].named_steps["prep"].transform(
    models["GradientBoosting (XGB proxy)"].named_steps["fe"].transform(X_val)
)
perm = permutation_importance(models["GradientBoosting (XGB proxy)"].named_steps["clf"], X_val_t, y_val, n_repeats=10, random_state=42, scoring="roc_auc")
gb_imp = pd.Series(perm.importances_mean, index=all_cols).sort_values(ascending=False)
print("\nTop 5 Boosted Model Features:\n", gb_imp.head(5).round(4))

# 4. Regression: Linear, Ridge, Lasso for loan_amount
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
r_cols = r_num + ord_cols + list(r_prep.fit(add_reg_features(X_r_tr)).named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols))

coef_table = pd.DataFrame(index=r_cols)
print("\n--- Regression Models ---")
for name, reg in [("Linear", LinearRegression()), ("Ridge", Ridge(alpha=10.0)), ("Lasso", Lasso(alpha=500.0, max_iter=5000))]:
    r_pipe = Pipeline([("fe", FunctionTransformer(add_reg_features, validate=False)), ("prep", r_prep), ("reg", reg)])
    r_pipe.fit(X_r_tr, y_r_tr)
    coef_table[name] = r_pipe.named_steps["reg"].coef_
    print(f"{name:8s} -> Val R2: {r2_score(y_r_va, r_pipe.predict(X_r_va)):.4f}")

print("Lasso zero coefficients count:", int(np.sum(np.isclose(coef_table["Lasso"], 0.0, atol=1e-6))))
print(coef_table.round(2))

km_cols = ["annual_income", "credit_score", "loan_amount"]
X_km = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[km_cols]))
inertias, sils = [], []

for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_km)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_km, labels))
    print(f"k={k} -> Silhouette: {sils[-1]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(2, 9), inertias, marker="o")
ax1.set_title("Elbow Curve")
ax1.set_xlabel("k")
ax1.set_ylabel("Inertia")
ax2.plot(range(2, 9), sils, marker="s")
ax2.set_title("Silhouette Scores")
ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette")
plt.tight_layout()
plt.savefig("task3_kmeans.png")
plt.show()
plt.close()