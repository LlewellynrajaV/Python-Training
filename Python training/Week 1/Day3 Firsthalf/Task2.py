import pandas as pd

data = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": [
        "Arun",
        "Priya",
        "Rahul",
        "Divya",
        "Karthik",
        "Meena",
        "Suresh",
        "Anitha",
    ],
    "department": ["Sales", "HR", "IT", "IT", "Finance", "Sales", "IT", "HR"],
    "salary": [30000, 45000, 60000, 75000, 90000, 52000, 68000, 48000],
    "city": [
        "Chennai",
        "Chennai",
        "Bangalore",
        "Chennai",
        "Mumbai",
        "Bangalore",
        "Chennai",
        "Mumbai",
    ],
}

df = pd.DataFrame(data)
df.to_csv("employees.csv", index=False)
df = pd.read_csv("employees.csv")

print("--- Head ---")
print(df.head())
print("\n--- Info ---")
df.info()
print("\n--- Describe ---")
print(df.describe())

print("\nName column:\n", df["name"])
print("\nName and Salary:\n", df[["name", "salary"]])

chennai_names = df.loc[df["city"] == "Chennai", "name"].tolist()
print("Chennai employees:", ", ".join(chennai_names))
print("Slice (iloc):\n", df.iloc[0:3, 1:3])

salary_above_50k = df[df["salary"] > 50000]
it_dept = df[df["department"] == "IT"]
print(f"Salary above 50000: {len(salary_above_50k)} employees")
print("IT department employees:", ", ".join(it_dept["name"].tolist()))