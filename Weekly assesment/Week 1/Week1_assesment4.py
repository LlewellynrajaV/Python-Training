import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI()

class EmployeeCreate(BaseModel):
    name: str = Field(min_length=1)
    department: str
    salary: float = Field(gt=0)

def load_employees():
    file_path = Path("employees.json")
    if file_path.exists():
        return json.loads(file_path.read_text(encoding="utf-8"))
    return []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/employees")
def get_employees():
    return load_employees()

@app.get("/employees/{name}")
def get_employee_by_name(name: str):
    employees = load_employees()
    for emp in employees:
        if emp["name"].lower() == name.lower():
            return emp
    raise HTTPException(status_code=404, detail="Employee not found")

@app.post("/employees")
def create_employee(employee: EmployeeCreate):
    employees = load_employees()
    new_emp = employee.model_dump()
    employees.append(new_emp)
    Path("employees.json").write_text(json.dumps(employees, indent=4), encoding="utf-8")
    return {
        "message": "Employee created",
        "employee": new_emp
    }

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_list_employees():
    res = client.get("/employees")
    assert res.status_code == 200

def test_employee_not_found():
    res = client.get("/employees/Ravi")
    assert res.status_code == 404

def test_create_employee_success():
    res = client.post("/employees", json={"name": "Kiran", "department": "IT", "salary": 55000})
    assert res.status_code == 200

def test_create_employee_invalid_salary():
    res = client.post("/employees", json={"name": "Kiran", "department": "IT", "salary": -5})
    assert res.status_code == 422

def test_create_employee_empty_name():
    res = client.post("/employees", json={"name": "", "department": "IT", "salary": 55000})
    assert res.status_code == 422

if __name__ == "__main__":
    import pytest

    print('GET /health -> {"status": "ok"}')
    print(f"GET /employees/Ravi -> {client.get('/employees/Ravi').status_code}")
    print(f"POST salary -5 -> {client.post('/employees', json={'name': 'Ravi', 'department': 'IT', 'salary': -5}).status_code}")

    pytest.main(["Week1_assesment4.py", "--cov=Week1_assesment4", "-q"])

    uvicorn.run("Week1_assesment4:app", host="127.0.0.1", port=5000, reload=True)