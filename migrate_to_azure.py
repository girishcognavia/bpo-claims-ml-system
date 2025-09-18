#!/usr/bin/env python3

import sqlite3
import pandas as pd
import pymssql
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

def migrate_data():
    """Migrate data from SQLite to Azure SQL"""

    # Read data from SQLite
    try:
        logger.info("Reading data from SQLite database...")
        sqlite_conn = sqlite3.connect('insurance_data.db')
        df = pd.read_sql_query("SELECT * FROM raw_applications", sqlite_conn)
        sqlite_conn.close()
        logger.info(f"Read {len(df)} records from SQLite")
        print(f"Data columns: {list(df.columns)}")
        print(f"Sample data:")
        print(df.head())

    except Exception as e:
        logger.error(f"Failed to read from SQLite: {e}")
        return False

    # Connect to Azure SQL
    try:
        logger.info("Connecting to Azure SQL...")
        azure_conn = pymssql.connect(
            server='serverica.database.windows.net',
            user='girish',
            password='Kumar@974',
            database='ICA'
        )

        cursor = azure_conn.cursor()

        # Create table if it doesn't exist
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='raw_applications' AND xtype='U')
        CREATE TABLE raw_applications (
            customer_id NVARCHAR(50),
            age INT,
            sex NVARCHAR(10),
            bmi FLOAT,
            children INT,
            smoker NVARCHAR(10),
            region NVARCHAR(20),
            charges FLOAT,
            medical_history NVARCHAR(200),
            employment_status NVARCHAR(50),
            annual_income INT,
            application_date DATETIME
        )
        """
        cursor.execute(create_table_sql)
        azure_conn.commit()
        logger.info("Table created/verified in Azure SQL")

        # Insert data in batches
        batch_size = 100
        total_inserted = 0

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]

            for _, row in batch.iterrows():
                cursor.execute("""
                    INSERT INTO raw_applications
                    (customer_id, age, sex, bmi, children, smoker, region, charges,
                     medical_history, employment_status, annual_income, application_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, tuple(row))

            azure_conn.commit()
            total_inserted += len(batch)
            logger.info(f"Inserted batch {i//batch_size + 1}, total records: {total_inserted}")

        # Verify data
        cursor.execute("SELECT COUNT(*) FROM raw_applications")
        count = cursor.fetchone()[0]
        logger.info(f"✅ Successfully migrated {count} records to Azure SQL")

        cursor.close()
        azure_conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to connect to Azure SQL: {e}")
        return False

if __name__ == "__main__":
    migrate_data()