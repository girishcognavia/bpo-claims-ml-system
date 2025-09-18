import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
from datetime import datetime
import time
import random
from faker import Faker
import os
from typing import Dict
import requests
from io import StringIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InsuranceDataIngestion:
    def __init__(self, connection_string: str):
        """Initialize the data ingestion service"""
        if not connection_string:
            raise ValueError("Connection string cannot be empty")

        self.connection_string = connection_string
        try:
            self.engine = create_engine(
                connection_string, pool_pre_ping=True, pool_recycle=3600, echo=False
            )
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise

        self.fake = Faker()

    def create_raw_applications_table(self) -> bool:
        """Create the raw_applications table if it doesn't exist"""
        try:
            logger.info("Creating raw_applications table...")

            create_table_sql = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='raw_applications' AND xtype='U')
            CREATE TABLE raw_applications (
                id INT IDENTITY(1,1) PRIMARY KEY,
                customer_id NVARCHAR(50),
                age INT,
                gender NVARCHAR(10),
                bmi FLOAT,
                children INT,
                smoker NVARCHAR(10),
                region NVARCHAR(20),
                charges FLOAT,
                medical_history NVARCHAR(200),
                employment_status NVARCHAR(50),
                annual_income INT,
                application_date DATETIME,
                created_date DATETIME DEFAULT GETDATE()
            );
            """

            with self.engine.begin() as conn:
                conn.execute(text(create_table_sql))

            logger.info("Raw applications table created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create raw_applications table: {e}")
            return False

    def download_public_dataset(self, timeout: int = 30) -> pd.DataFrame:
        """Download and prepare public insurance dataset"""
        try:
            url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
            logger.info(f"Attempting to download dataset from {url}")

            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                logger.warning("Downloaded dataset is empty")
                return self.generate_synthetic_data(1000)

            logger.info(f"Downloaded public dataset with {len(df)} records")
            return df

        except (
            requests.exceptions.RequestException,
            pd.errors.EmptyDataError,
            Exception,
        ) as e:
            logger.warning(f"Failed to download public dataset: {e}")
            return self.generate_synthetic_data(1000)

    def generate_synthetic_data(self, num_records: int = 100) -> pd.DataFrame:
        """Generate synthetic insurance application data"""
        logger.info(f"Generating {num_records} synthetic records")

        data = []
        for _ in range(num_records):
            # Generate realistic insurance application data
            age = random.randint(18, 80)
            gender = random.choice(["male", "female"])
            bmi = round(random.gauss(28, 6), 1)  # Normal distribution around 28
            children = random.randint(0, 5)
            smoker = random.choice(["yes", "no"])
            region = random.choice(["northeast", "northwest", "southeast", "southwest"])

            # Generate charges based on risk factors
            base_charge = 1000
            age_factor = age * 20
            bmi_factor = max(0, (bmi - 25) * 50)
            smoker_factor = 20000 if smoker == "yes" else 0
            children_factor = children * 500

            charges = (
                base_charge + age_factor + bmi_factor + smoker_factor + children_factor
            )
            charges += random.gauss(0, 2000)  # Add some noise
            charges = max(500, charges)  # Minimum charge

            # Additional fields for more realistic data
            employment_status = random.choice(
                ["employed", "unemployed", "self-employed", "retired"]
            )
            annual_income = random.randint(20000, 150000)

            medical_conditions = []
            if smoker == "yes":
                medical_conditions.append("smoker")
            if bmi > 30:
                medical_conditions.append("obesity")
            if age > 60:
                medical_conditions.append("senior")

            data.append(
                {
                    "customer_id": f"CUST_{self.fake.uuid4()[:8].upper()}",
                    "age": age,
                    "gender": gender,
                    "bmi": bmi,
                    "children": children,
                    "smoker": smoker,
                    "region": region,
                    "charges": round(charges, 2),
                    "medical_history": (
                        ", ".join(medical_conditions) if medical_conditions else "none"
                    ),
                    "employment_status": employment_status,
                    "annual_income": annual_income,
                    "application_date": self.fake.date_time_between(
                        start_date="-1y", end_date="now"
                    ),
                }
            )

        return pd.DataFrame(data)

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data"""
        logger.info("Validating data quality")

        initial_count = len(df)

        # Remove invalid records
        df_cleaned = df.dropna().copy()
        df_cleaned = df_cleaned[df_cleaned["age"].between(18, 100)]
        df_cleaned = df_cleaned[
            df_cleaned["bmi"].between(15, 50)
        ]  # More realistic BMI range
        df_cleaned = df_cleaned[df_cleaned["charges"] > 0]

        # Additional validation
        if "children" in df_cleaned.columns:
            df_cleaned = df_cleaned[df_cleaned["children"].between(0, 10)]

        # Validate data types
        numeric_columns = ["age", "bmi", "charges"]
        for col in numeric_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
                df_cleaned = df_cleaned.dropna(subset=[col]).copy()

        df = df_cleaned

        # Standardize categorical values
        if "gender" in df.columns:
            df["gender"] = df["gender"].str.lower()
        elif "sex" in df.columns:
            df["gender"] = df["sex"].str.lower()
            df = df.drop("sex", axis=1)

        if "smoker" in df.columns:
            df["smoker"] = df["smoker"].str.lower()

        if "region" in df.columns:
            df["region"] = df["region"].str.lower()

        final_count = len(df)
        logger.info(
            f"Data validation complete. {initial_count - final_count} records removed"
        )

        return df

    def insert_raw_data(self, df: pd.DataFrame) -> bool:
        """Insert data into raw_applications table"""
        if df.empty:
            logger.warning("Cannot insert empty dataframe")
            return False

        try:
            logger.info(f"Inserting {len(df)} records into raw_applications table")

            # Ensure required columns exist
            required_columns = [
                "customer_id",
                "age",
                "gender",
                "bmi",
                "children",
                "smoker",
                "region",
                "charges",
            ]

            for col in required_columns:
                if col not in df.columns:
                    if col == "customer_id":
                        df[col] = [f"CUST_{i:06d}" for i in range(len(df))]
                    else:
                        df[col] = None

            # Add missing columns with defaults
            if "medical_history" not in df.columns:
                df["medical_history"] = "none"
            if "employment_status" not in df.columns:
                df["employment_status"] = "unknown"
            if "annual_income" not in df.columns:
                df["annual_income"] = df["charges"] * 4  # Estimate income
            if "application_date" not in df.columns:
                df["application_date"] = datetime.now()

            # Insert data with transaction in batches (Azure SQL limit: 1000 rows)
            batch_size = 500
            with self.engine.begin() as conn:
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i : i + batch_size]
                    batch.to_sql(
                        "raw_applications", conn, if_exists="append", index=False
                    )
                    logger.info(
                        f"Inserted batch {i//batch_size + 1}: {len(batch)} records"
                    )
            logger.info("Data insertion successful")
            return True

        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            return False

    def simulate_real_time_ingestion(self, duration_minutes: int = 5):
        """Simulate real-time data ingestion"""
        logger.info(f"Starting real-time simulation for {duration_minutes} minutes")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while time.time() < end_time:
            # Generate small batch of new applications
            batch_size = random.randint(1, 5)
            new_data = self.generate_synthetic_data(batch_size)

            if self.insert_raw_data(new_data):
                logger.info(f"Inserted batch of {batch_size} records")

            # Wait before next batch
            time.sleep(random.randint(5, 15))

        logger.info("Real-time simulation completed")

    def get_ingestion_stats(self) -> Dict:
        """Get statistics about ingested data"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT
                        COUNT(*) as total_records,
                        MIN(application_date) as earliest_date,
                        MAX(application_date) as latest_date,
                        AVG(CAST(age AS FLOAT)) as avg_age,
                        AVG(charges) as avg_charges
                    FROM raw_applications
                """
                    )
                )

                row = result.fetchone()
                return {
                    "total_records": row[0],
                    "earliest_date": row[1],
                    "latest_date": row[2],
                    "avg_age": round(row[3], 1) if row[3] else 0,
                    "avg_charges": round(row[4], 2) if row[4] else 0,
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


def main():
    """Main execution function"""
    connection_string = os.getenv("AZURE_SQL_CONNECTION_STRING")

    if not connection_string:
        print("AZURE_SQL_CONNECTION_STRING not found, using SQLite for testing...")
        connection_string = "sqlite:///insurance_data.db"

    # Initialize ingestion service
    ingestion = InsuranceDataIngestion(connection_string)

    # Step 0: Create table if it doesn't exist
    print("Step 0: Creating database table...")
    if not ingestion.create_raw_applications_table():
        print("Failed to create table. Exiting.")
        return

    # Step 1: Generate 5000 synthetic records
    print("Step 1: Generating 5000 synthetic records...")
    df = ingestion.generate_synthetic_data(5000)

    # Step 2: Validate data
    print("Step 2: Validating data...")
    df = ingestion.validate_data(df)

    # Step 3: Load initial batch
    print("Step 3: Loading initial data...")
    if ingestion.insert_raw_data(df):
        print(f"Successfully loaded {len(df)} records")

    # Step 4: Show statistics
    stats = ingestion.get_ingestion_stats()
    print(f"Ingestion Statistics: {stats}")

    # Step 5: Simulate real-time ingestion (optional)
    simulate_realtime = input("Run real-time simulation? (y/n): ").lower() == "y"
    if simulate_realtime:
        duration = int(input("Duration in minutes (default 2): ") or "2")
        ingestion.simulate_real_time_ingestion(duration)


if __name__ == "__main__":
    main()
