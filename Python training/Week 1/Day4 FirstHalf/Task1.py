import matplotlib.pyplot as plt
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
pd.DataFrame(data).to_csv("employees.csv", index=False)

df = pd.read_csv("employees.csv")
df.columns = df.columns.str.strip()

avg_salary = df.groupby("department")["salary"].mean()
plt.figure(figsize=(8, 5))
bars = plt.bar(avg_salary.index, avg_salary.values, color="skyblue")
plt.title("Average Salary per Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
for bar in bars:
  yval = bar.get_height()
  plt.text(
      bar.get_x() + bar.get_width() / 2,
      yval + 1000,
      f"{yval:.0f}",
      ha="center",
      va="bottom",
  )
plt.savefig("avg_salary_bar.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(df["salary"], bins=5, color="orange", edgecolor="black")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.savefig("salary_histogram.png")
plt.close()

plt.figure(figsize=(8, 5))
for dept, group in df.groupby("department"):
  plt.scatter(group["id"], group["salary"], label=dept, s=100)
plt.title("Employee ID vs Salary by Department")
plt.xlabel("Employee ID")
plt.ylabel("Salary")
plt.legend()
plt.savefig("id_vs_salary_scatter.png")
plt.close()

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [12, 15, 14, 18, 22, 25]
plt.figure(figsize=(8, 5))
plt.plot(months, sales, marker="o", color="green", linewidth=2)
plt.title("Monthly Sales (Jan to Jun)")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.savefig("monthly_sales_line.png")
plt.close()

print("Success! All charts saved and generated.")