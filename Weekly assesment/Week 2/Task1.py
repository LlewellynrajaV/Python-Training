import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("loan_data.csv")
X = df.drop(columns=["default"])
y = df["default"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

num_cols = ["age", "annual_income", "credit_score", "debt_to_income", "employment_years", "loan_amount"]
imp = SimpleImputer(strategy="median")
X_tr = imp.fit_transform(X_train[num_cols])
X_va = imp.transform(X_val[num_cols])

depths = [1, 3, 5, 10, None]
train_accs = []
val_accs = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_tr, y_train)
    tr_acc = accuracy_score(y_train, tree.predict(X_tr))
    va_acc = accuracy_score(y_val, tree.predict(X_va))
    train_accs.append(tr_acc)
    val_accs.append(va_acc)
    print(f"Depth {d}: train {tr_acc:.2f} val {va_acc:.2f}")

plt.figure(figsize=(6, 4))
plt.plot([str(d) for d in depths], train_accs, marker="o", label="Train Accuracy")
plt.plot([str(d) for d in depths], val_accs, marker="s", label="Val Accuracy")
plt.title("Decision Tree Overfitting")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("task1_overfitting.png")
plt.show()
plt.close()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
scores = cross_val_score(lr_pipe, X_tr, y_train, cv=cv, scoring="accuracy")
print(f"CV: {scores.mean():.2f} +/- {scores.std():.2f}")

X_all_num = imp.fit_transform(X[num_cols])

X_scaled_all = StandardScaler().fit_transform(X_all_num)
X_tr_bad, X_va_bad, y_tr_bad, y_va_bad = train_test_split(X_scaled_all, y, test_size=0.20, random_state=42, stratify=y)
acc_bad = accuracy_score(y_va_bad, LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_bad, y_tr_bad).predict(X_va_bad))

X_tr_good, X_va_good, y_tr_good, y_va_good = train_test_split(X_all_num, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_good_s = scaler.fit_transform(X_tr_good)
X_va_good_s = scaler.transform(X_va_good)
acc_good = accuracy_score(y_va_good, LogisticRegression(max_iter=1000, random_state=42).fit(X_tr_good_s, y_tr_good).predict(X_va_good_s))

print(f"Leakage Val Acc: {acc_bad:.4f} | Correct Val Acc: {acc_good:.4f}")