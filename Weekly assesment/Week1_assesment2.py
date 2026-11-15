import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Union
import openpyxl
import pymupdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_sample_csv() -> None:
    csv_path = Path("employees.csv")
    if not csv_path.exists():
        sample_data = (
            "name,department,salary\n"
            "Arun,Sales,30000\n"
            "Priya,HR,45000\n"
            "Rahul,IT,60000\n"
            "Divya,IT,75000\n"
            "Karthik,Finance,90000\n"
        )
        csv_path.write_text(sample_data, encoding="utf-8")

def read_file(path: Union[str, Path]) -> Optional[str]:
    """Reads a file safely with exception handling and logging."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        print(f"ERROR File not found: {path}")
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in file: {path}")
    return None

def process_employee_files() -> None:
    csv_path = Path("employees.csv")
    json_path = Path("employees.json")
    high_csv_path = Path("high_earners.csv")

    rows: List[dict] = []
    if csv_path.exists():
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["salary"] = int(row["salary"])
                rows.append(row)

        json_path.write_text(json.dumps(rows, indent=4), encoding="utf-8")

        high_earners = [r for r in rows if r["salary"] > 50000]
        with open(high_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "department", "salary"])
            writer.writeheader()
            writer.writerows(high_earners)

        print(f"Rows read: {len(rows)}  High earners written: {len(high_earners)}")

def create_and_read_excel() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Employees"
    ws.append(["name", "department", "salary"])
    ws.append(["Arun", "Sales", 30000])
    ws.append(["Priya", "HR", 45000])
    ws.append(["Rahul", "IT", 60000])
    ws.append(["Divya", "IT", 75000])
    ws.append(["Karthik", "Finance", 90000])
    wb.save("salary.xlsx")

    wb_read = openpyxl.load_workbook("salary.xlsx")
    sheet = wb_read.active
    count = sum(1 for _ in sheet.iter_rows(min_row=2, values_only=True))
    print(f"salary.xlsx created and read back: {count} rows")

def extract_pdf_preview(pdf_path: str) -> None:
    """Extracts page 1 text of a PDF and prints the first 100 characters."""
    if Path(pdf_path).exists():
        doc = pymupdf.open(pdf_path)
        page_text = doc[0].get_text()
        print("PDF Page 1 (first 100 chars):", repr(page_text[:100]))

def find_employee(name: str) -> Optional[dict]:
    json_path = Path("employees.json")
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for emp in data:
            if emp["name"].lower() == name.lower():
                return emp
    return None

if __name__ == "__main__":
    setup_sample_csv()
    process_employee_files()
    create_and_read_excel()
    extract_pdf_preview("Week1_Assessment.pdf")
    read_file("missing.txt")