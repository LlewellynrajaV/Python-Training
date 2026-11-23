import functools

def calculate_bonus(salary: float, rate: float = 0.1, *extras: float, **options) -> float:
    bonus = salary * rate + sum(extras)
    if "cap" in options and bonus > options["cap"]:
        bonus = options["cap"]
    return bonus

class Employee:
    def __init__(self, name: str, salary: float):
        self.name = name
        self.salary = salary

    def get_details(self) -> str:
        return f"{self.name} {self.salary}"

    @property
    def annual_salary(self) -> float:
        return self.salary * 12

class Manager(Employee):
    def __init__(self, name: str, salary: float, team_size: int):
        super().__init__(name, salary)
        self.team_size = team_size

    def get_details(self) -> str:
        return f"{self.name} {self.salary}\nTeam Size: {self.team_size}"

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Calling get_details")
        return func(*args, **kwargs)
    return wrapper

Manager.get_details = log_call(Manager.get_details)

employees_data = [
    {"name": "Arun", "department": "Sales", "salary": 30000},
    {"name": "Priya", "department": "HR", "salary": 45000},
    {"name": "Rahul", "department": "IT", "salary": 60000},
    {"name": "Divya", "department": "IT", "salary": 75000},
    {"name": "Karthik", "department": "Finance", "salary": 90000}
]

for emp in employees_data:
    print(emp["name"])
    sal = emp["salary"]
    if sal < 50000:
        print("Low Salary")
    elif sal < 70000:
        print("Medium Salary")
    else:
        print("High Salary")

mgr = Manager("Divya", 75000, 5)
print(mgr.get_details())