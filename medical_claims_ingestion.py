#!/usr/bin/env python3

"""
Medical Claims Data Ingestion Service
Generates and ingests realistic medical claims data for claim rejection prediction
"""

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
from dotenv import load_dotenv
from medical_claims_schema import MedicalClaimsGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicalClaimsIngestion:
    def __init__(self, connection_string: str):
        """Initialize the medical claims ingestion service"""
        if not connection_string:
            raise ValueError("Connection string cannot be empty")

        self.connection_string = connection_string
        try:
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise

        self.claims_generator = MedicalClaimsGenerator()

    def create_claims_table(self) -> bool:
        """Create the medical_claims table if it doesn't exist"""
        try:
            logger.info("Creating medical_claims table...")

            create_table_sql = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='medical_claims' AND xtype='U')
            CREATE TABLE medical_claims (
                claim_id VARCHAR(50) PRIMARY KEY,
                patient_id VARCHAR(50) NOT NULL,
                provider_id VARCHAR(50) NOT NULL,

                -- Dates
                service_date DATETIME NOT NULL,
                submission_date DATETIME NOT NULL,
                days_to_submission INT,

                -- Patient demographics
                patient_age INT,
                patient_gender CHAR(1),
                patient_state VARCHAR(2),

                -- Policy details
                policy_type VARCHAR(10),
                deductible_amount DECIMAL(10,2),
                copay_amount DECIMAL(10,2),

                -- Provider information
                provider_specialty VARCHAR(50),
                provider_network_status VARCHAR(20),
                provider_years_experience INT,
                provider_monthly_claims INT,
                provider_avg_claim_amount DECIMAL(10,2),

                -- Clinical details
                primary_diagnosis_code VARCHAR(10),
                procedure_code VARCHAR(10),
                claim_type VARCHAR(20),

                -- Financial
                billed_amount DECIMAL(10,2),
                amount_per_age DECIMAL(10,2),

                -- Patient history
                patient_previous_claims INT,
                patient_previous_rejections INT,
                patient_rejection_rate DECIMAL(5,3),

                -- Documentation and quality
                documentation_quality_score DECIMAL(4,3),
                supporting_documents_count INT,

                -- Risk indicators
                is_weekend_submission BIT,
                is_after_hours_submission BIT,
                submission_hour INT,
                high_fraud_area BIT,

                -- Derived features
                billed_amount_zscore DECIMAL(10,6),
                provider_specialty_risk DECIMAL(5,3),

                -- Target variable
                is_rejected BIT NOT NULL,
                rejection_reason VARCHAR(100),
                risk_score DECIMAL(4,3),

                -- Audit fields
                created_date DATETIME DEFAULT GETDATE()
            );

            -- Create indexes for better performance
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_medical_claims_patient_id')
                CREATE INDEX idx_medical_claims_patient_id ON medical_claims (patient_id);

            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_medical_claims_provider_id')
                CREATE INDEX idx_medical_claims_provider_id ON medical_claims (provider_id);

            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_medical_claims_service_date')
                CREATE INDEX idx_medical_claims_service_date ON medical_claims (service_date);

            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_medical_claims_is_rejected')
                CREATE INDEX idx_medical_claims_is_rejected ON medical_claims (is_rejected);
            """

            with self.engine.begin() as conn:
                conn.execute(text(create_table_sql))

            logger.info("Medical claims table and indexes created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            return False

    def generate_claims_data(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate synthetic medical claims data"""
        logger.info(f"Generating {num_records} medical claims records")
        return self.claims_generator.generate_claims_data(num_records)

    def validate_claims_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the claims data"""
        logger.info("Validating claims data quality")

        initial_count = len(df)

        # Remove invalid records
        df_cleaned = df.dropna(subset=['claim_id', 'patient_id', 'provider_id']).copy()
        df_cleaned = df_cleaned[df_cleaned['patient_age'].between(0, 120)]
        df_cleaned = df_cleaned[df_cleaned['billed_amount'] > 0]
        df_cleaned = df_cleaned[df_cleaned['documentation_quality_score'].between(0, 1)]

        # Data type conversions
        df_cleaned['is_rejected'] = df_cleaned['is_rejected'].astype(bool)
        df_cleaned['is_weekend_submission'] = df_cleaned['is_weekend_submission'].astype(bool)
        df_cleaned['is_after_hours_submission'] = df_cleaned['is_after_hours_submission'].astype(bool)
        df_cleaned['high_fraud_area'] = df_cleaned['high_fraud_area'].astype(bool)

        final_count = len(df_cleaned)
        logger.info(
            f"Data validation complete. {initial_count - final_count} records removed"
        )

        return df_cleaned

    def insert_claims_data(self, df: pd.DataFrame) -> bool:
        """Insert claims data into medical_claims table"""
        if df.empty:
            logger.warning("Cannot insert empty dataframe")
            return False

        try:
            logger.info(f"Inserting {len(df)} claims records into medical_claims table")

            # Insert data with transaction in batches (Azure SQL limit: 1000 rows)
            batch_size = 500
            with self.engine.begin() as conn:
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    batch.to_sql("medical_claims", conn, if_exists="append", index=False)
                    logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")

            logger.info("Claims data insertion successful")
            return True

        except Exception as e:
            logger.error(f"Failed to insert claims data: {e}")
            return False

    def get_claims_statistics(self) -> Dict:
        """Get statistics about ingested claims data"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                    SELECT
                        COUNT(*) as total_claims,
                        SUM(CASE WHEN is_rejected = 1 THEN 1 ELSE 0 END) as rejected_claims,
                        AVG(CAST(patient_age AS FLOAT)) as avg_patient_age,
                        AVG(billed_amount) as avg_billed_amount,
                        AVG(risk_score) as avg_risk_score,
                        MIN(service_date) as earliest_service,
                        MAX(service_date) as latest_service,
                        COUNT(DISTINCT patient_id) as unique_patients,
                        COUNT(DISTINCT provider_id) as unique_providers
                    FROM medical_claims
                """)
                )

                row = result.fetchone()
                if row and row[0] > 0:
                    rejection_rate = (row[1] / row[0]) * 100 if row[0] > 0 else 0
                    return {
                        "total_claims": row[0],
                        "rejected_claims": row[1],
                        "rejection_rate_percent": round(rejection_rate, 2),
                        "avg_patient_age": round(row[2], 1) if row[2] else 0,
                        "avg_billed_amount": round(row[3], 2) if row[3] else 0,
                        "avg_risk_score": round(row[4], 3) if row[4] else 0,
                        "earliest_service": row[5],
                        "latest_service": row[6],
                        "unique_patients": row[7],
                        "unique_providers": row[8]
                    }
                else:
                    return {"total_claims": 0}

        except Exception as e:
            logger.error(f"Failed to get claims statistics: {e}")
            return {}

    def get_rejection_analysis(self) -> Dict:
        """Get detailed rejection analysis"""
        try:
            with self.engine.connect() as conn:
                # Rejection reasons analysis
                result = conn.execute(
                    text("""
                    SELECT
                        rejection_reason,
                        COUNT(*) as count,
                        AVG(billed_amount) as avg_amount
                    FROM medical_claims
                    WHERE is_rejected = 1 AND rejection_reason IS NOT NULL
                    GROUP BY rejection_reason
                    ORDER BY count DESC
                """)
                )

                rejection_reasons = []
                for row in result:
                    rejection_reasons.append({
                        "reason": row[0],
                        "count": row[1],
                        "avg_amount": round(row[2], 2) if row[2] else 0
                    })

                # Provider specialty analysis
                result = conn.execute(
                    text("""
                    SELECT
                        provider_specialty,
                        COUNT(*) as total_claims,
                        SUM(CASE WHEN is_rejected = 1 THEN 1 ELSE 0 END) as rejected_claims,
                        AVG(billed_amount) as avg_amount
                    FROM medical_claims
                    GROUP BY provider_specialty
                    ORDER BY total_claims DESC
                """)
                )

                specialty_analysis = []
                for row in result:
                    rejection_rate = (row[2] / row[1]) * 100 if row[1] > 0 else 0
                    specialty_analysis.append({
                        "specialty": row[0],
                        "total_claims": row[1],
                        "rejected_claims": row[2],
                        "rejection_rate": round(rejection_rate, 2),
                        "avg_amount": round(row[3], 2) if row[3] else 0
                    })

                return {
                    "rejection_reasons": rejection_reasons,
                    "specialty_analysis": specialty_analysis
                }

        except Exception as e:
            logger.error(f"Failed to get rejection analysis: {e}")
            return {}

    def simulate_real_time_claims(self, duration_minutes: int = 5):
        """Simulate real-time claims ingestion"""
        logger.info(f"Starting real-time claims simulation for {duration_minutes} minutes")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while time.time() < end_time:
            # Generate small batch of new claims
            batch_size = random.randint(5, 20)
            new_claims = self.generate_claims_data(batch_size)
            validated_claims = self.validate_claims_data(new_claims)

            if self.insert_claims_data(validated_claims):
                logger.info(f"Inserted batch of {len(validated_claims)} claims")

            # Wait before next batch
            time.sleep(random.randint(10, 30))

        logger.info("Real-time claims simulation completed")


def main():
    """Main execution function"""
    connection_string = os.getenv("AZURE_SQL_CONNECTION_STRING")

    if not connection_string:
        print("AZURE_SQL_CONNECTION_STRING not found, using SQLite for testing...")
        connection_string = "sqlite:///medical_claims.db"

    # Initialize claims ingestion service
    ingestion = MedicalClaimsIngestion(connection_string)

    # Step 1: Create table
    print("Step 1: Creating medical claims table...")
    if not ingestion.create_claims_table():
        print("Failed to create table. Exiting.")
        return

    # Step 2: Generate claims data
    print("Step 2: Generating medical claims data...")
    num_claims = int(input("Number of claims to generate (default 5000): ") or "5000")
    claims_df = ingestion.generate_claims_data(num_claims)

    # Step 3: Validate data
    print("Step 3: Validating claims data...")
    claims_df = ingestion.validate_claims_data(claims_df)

    # Step 4: Load data
    print("Step 4: Loading claims data...")
    if ingestion.insert_claims_data(claims_df):
        print(f"Successfully loaded {len(claims_df)} claims")

    # Step 5: Show statistics
    print("Step 5: Analyzing claims data...")
    stats = ingestion.get_claims_statistics()
    print(f"Claims Statistics: {stats}")

    # Step 6: Show rejection analysis
    rejection_analysis = ingestion.get_rejection_analysis()
    if rejection_analysis:
        print("\nTop Rejection Reasons:")
        for reason in rejection_analysis.get('rejection_reasons', [])[:5]:
            print(f"  - {reason['reason']}: {reason['count']} claims")

        print("\nSpecialty Rejection Rates:")
        for specialty in rejection_analysis.get('specialty_analysis', [])[:5]:
            print(f"  - {specialty['specialty']}: {specialty['rejection_rate']}% ({specialty['rejected_claims']}/{specialty['total_claims']})")

    # Step 7: Real-time simulation (optional)
    simulate_realtime = input("\nRun real-time claims simulation? (y/n): ").lower() == "y"
    if simulate_realtime:
        duration = int(input("Duration in minutes (default 2): ") or "2")
        ingestion.simulate_real_time_claims(duration)


if __name__ == "__main__":
    main()