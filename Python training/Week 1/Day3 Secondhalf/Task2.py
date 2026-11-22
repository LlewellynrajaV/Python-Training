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
df = df.drop_duplicates()
df["amount"] = df["amount"].fillna(df["amount"].median())

customers = pd.DataFrame(
    {
        "customer_id": [101, 102, 103, 105],
        "name": ["Arun", "Priya", "Rahul", "Divya"],
        "city": ["Chennai", "Mumbai", "Chennai", "Bangalore"],
    }
)

inner_join_df = pd.merge(df, customers, on="customer_id", how="inner")
print(f"Inner join rows: {len(inner_join_df)} (customer 104 dropped)")

left_join_df = pd.merge(customers, df, on="customer_id", how="left")
no_orders_df = left_join_df[left_join_df["amount"].isna()]
no_orders_names = ", ".join(no_orders_df["name"].tolist())
print(f"No orders: {no_orders_names}")

outer_join_df = pd.merge(customers, df, on="customer_id", how="outer")
orphan_orders = outer_join_df[outer_join_df["name"].isna()][
    "customer_id"
].values
print(f"Order without customer: customer_id {int(orphan_orders[0])}")