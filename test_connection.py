#!/usr/bin/env python3

import pymssql
from dotenv import load_dotenv
import os

load_dotenv()

# Test direct connection
try:
    conn = pymssql.connect(
        server='serverica.database.windows.net',
        user='girish',
        password='girish@974',
        database='ICA',
        port=1433
    )
    print("✅ Direct connection successful!")

    cursor = conn.cursor()
    cursor.execute("SELECT @@VERSION")
    result = cursor.fetchone()
    print(f"SQL Server Version: {result[0][:50]}...")

    # Test if table exists
    cursor.execute("""
        IF OBJECT_ID('raw_applications', 'U') IS NOT NULL
            SELECT 1 as table_exists
        ELSE
            SELECT 0 as table_exists
    """)
    table_exists = cursor.fetchone()[0]
    print(f"Table 'raw_applications' exists: {'Yes' if table_exists else 'No'}")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"❌ Connection failed: {e}")