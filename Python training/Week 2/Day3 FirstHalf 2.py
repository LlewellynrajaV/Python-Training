import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

df_num = pd.DataFrame({
    'age': [25, np.nan, 40, 35, np.nan],
    'income': [30000, 45000, np.nan, 80000, 52000],
})
df_num[['age', 'income']] = KNNImputer(n_neighbors=2).fit_transform(df_num)
print('=== Step 1: Base Numeric DataFrame ===')
print(df_num, '\n')

std_scaler = StandardScaler()
df_std = pd.DataFrame(
    std_scaler.fit_transform(df_num), columns=df_num.columns, index=df_num.index
)
print('=== Step 2: After StandardScaler ===')
print(df_std)
print('Mean:', np.round(df_std.mean().values, 6))
print('Std (ddof=0):', np.round(df_std.std(ddof=0).values, 6), '\n')

minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(df_num),
    columns=df_num.columns,
    index=df_num.index,
)
print('=== Step 3: After MinMaxScaler ===')
print(df_minmax)
print('Min:', df_minmax.min().values)
print('Max:', df_minmax.max().values, '\n')

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit ONLY on X_train
X_test_scaled = scaler.transform(X_test)  # transform X_test only

knn_unscaled = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn_scaled = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)

dt_unscaled = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
dt_scaled = DecisionTreeClassifier(random_state=42).fit(X_train_scaled, y_train)

results_df = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree'],
    'Accuracy_Without_Scaling': [
        accuracy_score(y_test, knn_unscaled.predict(X_test)),
        accuracy_score(y_test, dt_unscaled.predict(X_test)),
    ],
    'Accuracy_With_Scaling': [
        accuracy_score(y_test, knn_scaled.predict(X_test_scaled)),
        accuracy_score(y_test, dt_scaled.predict(X_test_scaled)),
    ],
})
print('=== Step 4: Model Comparison (Breast Cancer Dataset) ===')
print(results_df.to_string(index=False))
