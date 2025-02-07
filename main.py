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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Regex-based validation functions
def validate_phone_number(value: Optional[str]) -> Optional[str]:
    """Validates phone numbers with a balance between strictness and flexibility."""
    if value:
        pattern = re.compile(r"^\+?[0-9\s\-\(\)]{7,20}$")  # Allows country codes, spaces, dashes, and parentheses
        if not pattern.match(value):
            return None  # Invalid format
    return value

def validate_linkedin_url(value: Optional[str]) -> Optional[str]:
    """Ensures LinkedIn link contains 'linkedin.com'."""
    if value and "linkedin.com" not in value:
        return None
    return value

def validate_x_twitter_url(value: Optional[str]) -> Optional[str]:
    """Ensures Twitter/X link contains 'twitter.com' or 'x.com'."""
    if value and not any(domain in value for domain in ["twitter.com", "x.com"]):
        return None
    return value

# Define Pydantic model for structured data extraction
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

class PartialCompanyList(BaseModel):
    """Pydantic Model that accepts valid structured data."""
    companies: List[Company]

def extract_data_from_batch(rows):
    """Extracts structured information from a batch of CSV rows using Ollama."""
    raw_data = "\n".join([", ".join(row) for row in rows])

    prompt = f"""
    You are an intelligent data extractor. Convert the given structured text into JSON format, 
    ensuring the values match the expected types.

    **Strictly follow this format:**
    {{
      "Company_Name": "Company name",
      "Company_Website": "Company website URL",
      "Company_Location": "Company location",
      "Company_Phone_Number": "Company phone number (must be valid)",
      "Email": "Company email (must be valid)",
      "LinkedIn_Link": "LinkedIn profile URL (must contain linkedin.com)",
      "Sector": "Industry sector",
      "Ticket_Size": "Investment ticket size (e.g., $100K, ₹50 Cr, €500K)",
      "X_Twitter_Account_Link": "Twitter/X profile URL (must contain twitter.com or x.com)",
      "Funding_Round": "Investment round (e.g., Seed, Series A, IPO)",
      "Individual_or_Corporation": "Specify if the entity is an individual or a corporation"
    }}

    ❗ **Important Instructions:**
    - **Do not change field names**; output must match the expected structure.
    - **Leave fields blank if data is missing.** Do not add extra fields.
    - **Ensure data is properly formatted (URLs, emails, phone numbers, currency).**
    - **Remove duplicates and irrelevant text.**

    **Extracted Data:**
    {raw_data}
    """

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model="llama3.2",
        format=PartialCompanyList.model_json_schema(),
    )

    print("\n🔍 Ollama Raw Response:\n", response.message.content, "\n")

    return PartialCompanyList.model_validate_json(response.message.content)

@app.command()
def process_csv(
    input_csv: Path = typer.Argument(..., help="Path to input CSV file"),
    output_csv: Path = typer.Argument(..., help="Path to output CSV file"),
    batch_size: int = typer.Option(5, "--batch-size", "-b", help="Number of rows per request"),
    error_log: Path = typer.Option("error_log.txt", "--error-log", "-e", help="Path to error log file")
):
    """
    Reads a CSV file, extracts structured data using Ollama in batches, 
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

        rows_buffer = []
        total_rows = sum(1 for _ in csv_reader)  # Count rows for tqdm
        csvfile.seek(0)  # Reset file pointer
        next(csv_reader)  # Skip header again

        with tqdm(total=total_rows, desc="Processing Rows", unit="row") as pbar:
            def process_batch(rows):
                """Processes a batch of rows."""
                try:
                    structured_data_batch = extract_data_from_batch(rows)

                    for structured_data in structured_data_batch.companies:
                        valid_data = {k: v for k, v in structured_data.model_dump().items() if v not in [None, "", "None"]}

                        if valid_data:
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
                except ValidationError as e:
                    log_error(f"Skipping batch due to validation error: {e}")
                except Exception as e:
                    log_error(f"Skipping batch due to unexpected error: {e}")

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for row in csv_reader:
                    if not any(row):
                        continue
                    rows_buffer.append(row)
                    if len(rows_buffer) >= batch_size:
                        futures.append(executor.submit(process_batch, rows_buffer))
                        rows_buffer = []
                    pbar.update(1)

                # Process any remaining rows
                if rows_buffer:
                    futures.append(executor.submit(process_batch, rows_buffer))
                    pbar.update(len(rows_buffer))

                # Wait for all futures to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except ValidationError as e:
                        log_error(f"Error processing batch: {e}")
                    except Exception as e:
                        log_error(f"Error processing batch: {e}")

    # Save extracted data to a CSV file
    conn.execute(f"COPY companies TO '{output_csv}' (HEADER, DELIMITER ',')")

    print(f"\n✅ Extraction complete! Data saved to {output_csv}")

if __name__ == "__main__":
    app()
