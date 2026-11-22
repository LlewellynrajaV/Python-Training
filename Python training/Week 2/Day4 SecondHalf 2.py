import time
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

X, y = make_classification(n_samples=2000, n_features=15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4]
}

start_time = time.time()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring="f1")
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print("--- GridSearchCV Results ---")
print("Best Params:", grid_search.best_params_)
print("Best F1 Score:", round(grid_search.best_score_, 4))
print("Time Taken:", round(grid_time, 2), "seconds")

grid_results_df = pd.DataFrame(grid_search.cv_results_)
print("\nGridSearch CV Results DataFrame:")
print(grid_results_df[["params", "mean_test_score", "rank_test_score"]].head())

param_dist = {
    "n_estimators": [50, 100, 150, 200, 250, 300],
    "max_depth": [None, 5, 10, 15, 20, 25, 30],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 8, 10]
}

start_time = time.time()
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42
)
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print("\n--- RandomizedSearchCV Results ---")
print("Best Params:", random_search.best_params_)
print("Best F1 Score:", round(random_search.best_score_, 4))
print("Time Taken:", round(random_time, 2), "seconds")

random_results_df = pd.DataFrame(random_search.cv_results_)
print("\nRandomizedSearch CV Results DataFrame:")
print(random_results_df[["params", "mean_test_score", "rank_test_score"]].head())

best_model = random_search.best_estimator_
y_test_pred = best_model.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred)

print("\nFinal Test Set F1 Score:", round(test_f1, 4))