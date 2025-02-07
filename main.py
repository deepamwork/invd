import json
import csv
import re
import duckdb
import typer
import logging
from tqdm import tqdm
from ollama import chat
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, HttpUrl, EmailStr, field_validator, ValidationError
from datetime import datetime

app = typer.Typer()

# Configure logging
logging.basicConfig(
    filename="error_log.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_error(message):
    """Logs errors with timestamps."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.error(f"{timestamp} - {message}")

class Company(BaseModel):
    Company_Name: str
    Company_Website: Optional[HttpUrl] = None
    Company_Location: Optional[str] = None
    Company_Phone_Number: Optional[str] = None
    Email: Optional[EmailStr] = None
    LinkedIn_Link: Optional[HttpUrl] = None
    Sector: Optional[str] = None
    Ticket_Size: Optional[str] = None
    X_Twitter_Account_Link: Optional[HttpUrl] = None
    Funding_Round: Optional[str] = None
    Individual_or_Corporation: Optional[str] = None

    @field_validator("Company_Phone_Number")
    @classmethod
    def validate_phone_number(cls, value):
        """Validates and auto-fixes phone numbers."""
        if value:
            value = re.sub(r"[^\d+]", "", value)  # Remove non-numeric characters except '+'
            pattern = re.compile(r"^\+?[0-9]{7,15}$")  # Allows 7-15 digits
            if not pattern.match(value):
                raise ValueError("Invalid phone number format")
        return value

    @field_validator("LinkedIn_Link")
    @classmethod
    def validate_linkedin_url(cls, value):
        """Ensures LinkedIn URL contains 'linkedin.com'."""
        if value and "linkedin.com" not in str(value):
            raise ValueError("Invalid LinkedIn URL. Must contain 'linkedin.com'")
        return value

    @field_validator("X_Twitter_Account_Link")
    @classmethod
    def validate_x_twitter_url(cls, value):
        """Ensures Twitter/X URL contains 'twitter.com' or 'x.com'."""
        if value and not any(domain in str(value) for domain in ["twitter.com", "x.com"]):
            raise ValueError("Invalid Twitter/X URL. Must contain 'twitter.com' or 'x.com'")
        return value

    @field_validator("Ticket_Size")
    @classmethod
    def validate_ticket_size(cls, value):
        """Ensures ticket size is in a valid format (e.g., $1M, ₹50 Cr, €500K)."""
        if value:
            pattern = re.compile(r"^[\$€₹]?[0-9]+(\.[0-9]+)?\s?(M|K|Cr|L|million|thousand)?$", re.IGNORECASE)
            if not pattern.match(value):
                raise ValueError("Invalid Ticket Size format")
        return value

class CompanyList(BaseModel):
    companies: List[Company]

def extract_data_from_row(row) -> Dict[str, Any]:
    """Extracts structured information for a single CSV row using Ollama and Pydantic."""
    raw_data = ", ".join(row)

    prompt = f"""
    Extract the following structured information from the given data.
    If a column is invalid or missing, leave it blank.

    Headers: "Company Name", "Company Website", "Company Location", "Company Phone Number", "Email",
    "LinkedIn Link", "Sector", "Ticket Size", "X Twitter Account Link", "Funding Round", "Individual or Corporation"

    Data:
    {raw_data}
    """

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model="llama3.2",  
        format=CompanyList.model_json_schema(),  # Request structured JSON output
    )

    # 🛑 Debug: Print the raw response from Ollama
    print("\n🔍 Ollama Raw Response:\n", response.message.content, "\n")

    structured_data = CompanyList.model_validate_json(response.message.content)

    # Convert Pydantic model into a dictionary
    return structured_data.companies[0].model_dump()

@app.command()
def process_csv(
    input_csv: Path = typer.Argument(..., help="Path to input CSV file"),
    output_csv: Path = typer.Argument(..., help="Path to output CSV file"),
    error_log: Path = typer.Option("error_log.txt", "--error-log", "-e", help="Path to error log file")
):
    """
    Reads a CSV file, extracts structured data using Ollama, 
    and stores results in DuckDB before saving to a new CSV file.
    Only valid fields are saved; invalid ones are left empty.
    """
    conn = duckdb.connect(database=':memory:')  # In-memory DuckDB

    # Create a DuckDB table
    conn.execute("""
        CREATE TABLE companies (
            Company_Name TEXT, Company_Website TEXT, Company_Location TEXT, Company_Phone_Number TEXT, 
            Email TEXT, LinkedIn_Link TEXT, Sector TEXT, Ticket_Size TEXT, 
            X_Twitter_Account_Link TEXT, Funding_Round TEXT, Individual_or_Corporation TEXT
        )
    """)

    with open(input_csv, newline='', encoding='utf-8') as csvfile, open(error_log, "w") as err_log:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)  # Skip header row

        total_rows = sum(1 for _ in csv_reader)  # Count rows for tqdm
        csvfile.seek(0)  # Reset file pointer
        next(csv_reader)  # Skip header again

        with tqdm(total=total_rows, desc="Processing Rows", unit="row") as pbar:
            for row in csv_reader:
                if not any(row):  # Skip empty rows
                    pbar.update(1)
                    continue

                try:
                    structured_data = extract_data_from_row(row)

                    structured_data["Company_Phone_Number"] = validate_phone_number(structured_data.get("Company_Phone_Number"))
                    structured_data["LinkedIn_Link"] = validate_linkedin_url(structured_data.get("LinkedIn_Link"))
                    structured_data["X_Twitter_Account_Link"] = validate_x_twitter_url(structured_data.get("X_Twitter_Account_Link"))

                    valid_data = {k: v for k, v in structured_data.items() if v not in [None, "", "None"]}

                    if valid_data:  # Only insert if valid data exists
                        conn.execute("""
                            INSERT INTO companies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            valid_data.get("Company_Name", ""),
                            valid_data.get("Company_Website", ""),
                            valid_data.get("Company_Location", ""),
                            valid_data.get("Company_Phone_Number", ""),
                            valid_data.get("Email", ""),
                            valid_data.get("LinkedIn_Link", ""),
                            valid_data.get("Sector", ""),
                            valid_data.get("Ticket_Size", ""),
                            valid_data.get("X_Twitter_Account_Link", ""),
                            valid_data.get("Funding_Round", ""),
                            valid_data.get("Individual_or_Corporation", ""),
                        ])
                except Exception as e:
                    log_error(f"Skipping row due to validation error: {e}\n")

                pbar.update(1)

    # Save extracted data to a CSV file
    conn.execute(f"COPY companies TO '{output_csv}' (HEADER, DELIMITER ',')")

    print(f"\n✅ Extraction complete! Data saved to {output_csv}")

if __name__ == "__main__":
    app()
