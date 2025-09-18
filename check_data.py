#!/usr/bin/env python3

import pymssql
from dotenv import load_dotenv
import os

load_dotenv()

def check_database():
    """Check what's actually in the Azure SQL database"""
    try:
        # Connect to Azure SQL
        conn = pymssql.connect(
            server='serverica.database.windows.net',
            user='girish',
            password='Krishna@974',
            database='ICA'
        )

        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = 'raw_applications'
        """)
        table_exists = cursor.fetchone()[0]
        print(f"Table 'raw_applications' exists: {'Yes' if table_exists else 'No'}")

        if table_exists:
            # Get table structure
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = 'raw_applications'
                ORDER BY ORDINAL_POSITION
            """)
            columns = cursor.fetchall()
            print(f"\nTable columns ({len(columns)}):")
            for col in columns:
                print(f"  - {col[0]} ({col[1]})")

            # Get record count
            cursor.execute("SELECT COUNT(*) FROM raw_applications")
            count = cursor.fetchone()[0]
            print(f"\nTotal records in table: {count}")

            if count > 0:
                # Show sample data
                cursor.execute("SELECT TOP 5 * FROM raw_applications")
                rows = cursor.fetchall()
                print(f"\nSample records:")
                for i, row in enumerate(rows, 1):
                    print(f"  Record {i}: {row}")

                # Show date range
                cursor.execute("""
                    SELECT
                        MIN(application_date) as earliest,
                        MAX(application_date) as latest
                    FROM raw_applications
                """)
                date_range = cursor.fetchone()
                print(f"\nDate range: {date_range[0]} to {date_range[1]}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_database()