#!/usr/bin/env python3
"""
Script to create sample SQLite database for Text-to-SQL case study.

Run this script to generate sample_database.db from schema.sql
"""

import sqlite3
import os

def create_database():
    """Create SQLite database from schema.sql file."""

    # Get directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # File paths
    schema_file = os.path.join(script_dir, "schema.sql")
    db_file = os.path.join(script_dir, "sample_database.db")

    # Remove existing database if it exists
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing database: {db_file}")

    # Create new database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Read and execute schema file
    with open(schema_file, 'r') as f:
        schema_sql = f.read()

    cursor.executescript(schema_sql)
    conn.commit()

    # Verify tables were created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    print(f"\n✅ Database created successfully: {db_file}")
    print(f"\nTables created: {len(tables)}")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"  - {table[0]}: {count} rows")

    conn.close()

    print("\n✨ Database is ready to use!")
    print(f"   Location: {db_file}")

if __name__ == "__main__":
    create_database()
