import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

data = pd.read_csv("loan_data.csv")
new_3_rows = data.drop(columns=["default"]).iloc[:3].copy()
df = data.iloc[3:].reset_index(drop=True)

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

knn_pipe = Pipeline([
    ("fe", FunctionTransformer(add_features, validate=False)),
    ("sel", FunctionTransformer(lambda d: d[num_cols], validate=False)),
    ("knn", KNNImputer(n_neighbors=5)),
])
knn_imputed = knn_pipe.fit_transform(X_train)
print("NaNs left after KNNImputer:", np.isnan(knn_imputed).sum())

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

ord_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(
        categories=[["High School", "Bachelor", "Master", "PhD"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("ord", ord_transformer, ord_cols),
    ("cat", cat_transformer, cat_cols),
])

pipeline = Pipeline([
    ("fe", FunctionTransformer(add_features, validate=False)),
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
print(f"Pipeline CV accuracy: {cv_scores.mean():.2f}")

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "loan_pipeline.joblib")

loaded_pipeline = joblib.load("loan_pipeline.joblib")
preds = loaded_pipeline.predict(new_3_rows)
print("Predictions for 3 new rows:", preds)