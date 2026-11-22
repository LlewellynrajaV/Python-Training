import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "max_depth": [2, 3, 4, 6],
    "n_estimators": [50, 100, 150, 200],
}

xgb_base = XGBClassifier(
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

grid_search.fit(X_train_val, y_train_val)

print("Best Parameters:", grid_search.best_params_)
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_
test_preds = best_xgb.predict(X_test)
print(f"Held-out Test Accuracy: {accuracy_score(y_test, test_preds):.4f}")