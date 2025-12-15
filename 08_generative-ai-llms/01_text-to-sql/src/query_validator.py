"""
Query Validator for Text-to-SQL System

This module validates and executes SQL queries safely.
"""

import sqlite3
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DML
from typing import Dict, List, Optional, Tuple
import re


class QueryValidator:
    """
    Validates and executes SQL queries with security checks.

    This class provides methods to:
    - Validate SQL syntax
    - Check for dangerous operations
    - Verify table/column existence
    - Execute queries safely
    """

    # Dangerous SQL keywords that should not be allowed
    DANGEROUS_KEYWORDS = {
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
        'TRUNCATE', 'REPLACE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE'
    }

    def __init__(self, database_path: str, read_only: bool = True):
        """
        Initialize query validator.

        Args:
            database_path: Path to SQLite database
            read_only: Whether to enforce read-only mode (only SELECT)
        """
        self.database_path = database_path
        self.read_only = read_only
        self._connection = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (create if needed)."""
        if self._connection is None:
            if self.read_only:
                # Open in read-only mode
                uri = f"file:{self.database_path}?mode=ro"
                self._connection = sqlite3.connect(uri, uri=True)
            else:
                self._connection = sqlite3.connect(self.database_path)

            self._connection.row_factory = sqlite3.Row
        return self._connection

    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query.

        Performs multiple validation checks:
        1. Syntax validation
        2. Security checks (no dangerous operations)
        3. Table/column existence checks

        Args:
            sql: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
            If valid: (True, None)
            If invalid: (False, error_message)
        """
        if not sql or not sql.strip():
            return False, "SQL query is empty"

        # 1. Parse SQL
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False, "Could not parse SQL query"
            statement = parsed[0]
        except Exception as e:
            return False, f"SQL parse error: {str(e)}"

        # 2. Check for dangerous keywords
        if self.read_only:
            is_safe, error = self._check_dangerous_keywords(statement)
            if not is_safe:
                return False, error

        # 3. Extract and validate tables/columns
        is_valid, error = self._validate_tables_columns(sql)
        if not is_valid:
            return False, error

        return True, None

    def _check_dangerous_keywords(
        self,
        statement: Statement
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for dangerous SQL keywords.

        Args:
            statement: Parsed SQL statement

        Returns:
            Tuple of (is_safe, error_message)
        """
        # Get all tokens
        tokens = list(statement.flatten())

        for token in tokens:
            if token.ttype in (Keyword, DML):
                keyword = token.value.upper()
                if keyword in self.DANGEROUS_KEYWORDS:
                    return False, f"Dangerous operation not allowed: {keyword}"

        # Additional check: ensure it starts with SELECT
        first_keyword = statement.get_type()
        if first_keyword != 'SELECT':
            return False, f"Only SELECT queries allowed (got {first_keyword})"

        return True, None

    def _validate_tables_columns(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that tables and columns exist.

        Args:
            sql: SQL query

        Returns:
            Tuple of (is_valid, error_message)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get all table names in database
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        valid_tables = {row[0].lower() for row in cursor.fetchall()}

        # Extract table names from SQL
        table_names = self._extract_table_names(sql)

        # Check if all tables exist
        for table in table_names:
            if table.lower() not in valid_tables:
                return False, f"Table '{table}' does not exist"

        # TODO: Could add column validation here, but it's complex
        # due to aliases, functions, etc. Let the dry_run catch column errors.

        return True, None

    def _extract_table_names(self, sql: str) -> List[str]:
        """
        Extract table names from SQL query.

        Args:
            sql: SQL query

        Returns:
            List of table names
        """
        tables = set()

        # Simple regex patterns to find table names
        # This is not perfect but works for most cases

        # FROM clause
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        from_matches = re.findall(from_pattern, sql, re.IGNORECASE)
        tables.update(from_matches)

        # JOIN clauses
        join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
        tables.update(join_matches)

        return list(tables)

    def dry_run(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Execute query with LIMIT 1 to test validity.

        This is a final check that the query actually runs.

        Args:
            sql: SQL query

        Returns:
            Tuple of (can_execute, error_message)
        """
        # Add LIMIT 1 if not present
        if 'LIMIT' not in sql.upper():
            test_sql = f"{sql.rstrip(';')} LIMIT 1"
        else:
            test_sql = sql

        try:
            conn = self._get_connection()
            cursor = conn.execute(test_sql)
            cursor.fetchall()
            return True, None
        except Exception as e:
            return False, str(e)

    def execute(
        self,
        sql: str,
        limit: int = 1000,
        timeout: float = 30.0
    ) -> Dict:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query to execute
            limit: Maximum number of rows to return
            timeout: Query timeout in seconds

        Returns:
            Dictionary with query results:
            {
                'columns': List of column names,
                'rows': List of row dictionaries,
                'row_count': Number of rows returned,
                'truncated': Whether results were truncated
            }

        Raises:
            ValueError: If SQL is invalid
            RuntimeError: If query execution fails
        """
        # Validate first
        is_valid, error_msg = self.validate(sql)
        if not is_valid:
            raise ValueError(f"Invalid SQL: {error_msg}")

        try:
            conn = self._get_connection()

            # Set timeout
            conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

            # Execute query
            cursor = conn.execute(sql)

            # Fetch results (up to limit)
            rows = cursor.fetchmany(limit + 1)  # Fetch one extra to check truncation

            # Convert to list of dicts
            columns = [desc[0] for desc in cursor.description]
            truncated = len(rows) > limit

            if truncated:
                rows = rows[:limit]

            row_dicts = [dict(zip(columns, row)) for row in rows]

            return {
                'columns': columns,
                'rows': row_dicts,
                'row_count': len(row_dicts),
                'truncated': truncated
            }

        except sqlite3.Error as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")

    def explain_query(self, sql: str) -> Dict:
        """
        Get query execution plan.

        Args:
            sql: SQL query

        Returns:
            Dictionary with execution plan information
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(f"EXPLAIN QUERY PLAN {sql}")
            plan_rows = cursor.fetchall()

            plan = [
                {
                    'id': row[0],
                    'parent': row[1],
                    'detail': row[3]
                }
                for row in plan_rows
            ]

            return {
                'plan': plan,
                'plan_text': '\n'.join(row['detail'] for row in plan)
            }

        except Exception as e:
            return {
                'error': str(e)
            }

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison.

    Removes whitespace, comments, and converts to uppercase for
    easy comparison of SQL queries.

    Args:
        sql: SQL query

    Returns:
        Normalized SQL string
    """
    # Parse and format
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql.strip().upper()

    # Format with consistent style
    formatted = sqlparse.format(
        sql,
        keyword_case='upper',
        identifier_case='lower',
        strip_comments=True,
        reindent=False
    )

    # Remove extra whitespace
    normalized = ' '.join(formatted.split())

    return normalized
