import pandas as pd

csv_content = """order_id,customer_id, amount
1,101,500
2,102,
3,101,700
3,101,700
4,103,300
5,104,900"""

with open("orders.csv", "w") as f:
    f.write(csv_content)

df = pd.read_csv("orders.csv")

df.columns = df.columns.str.strip()

print("Missing values:\n", df.isna().sum())
print("Duplicates count:", df.duplicated().sum())

df = df.drop_duplicates()
median_amount = df["amount"].median()
print(f"Median used: {median_amount}")
df["amount"] = df["amount"].fillna(median_amount)

df["amount"] = df["amount"].astype(float)
df["order_id"] = df["order_id"].astype(str)

agg_result = df.groupby("customer_id").agg(
    total=("amount", "sum"), orders=("order_id", "count")
)
print("\nAggregated Result:\n", agg_result)