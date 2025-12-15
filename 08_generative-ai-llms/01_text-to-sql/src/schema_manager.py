"""
Schema Manager for Text-to-SQL System

This module handles database schema extraction, caching, and formatting
for LLM consumption.
"""

import sqlite3
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from functools import lru_cache
import re


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    type: str
    nullable: bool
    primary_key: bool
    default_value: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    columns: List[ColumnInfo]
    foreign_keys: List[Tuple[str, str, str]]  # (column, ref_table, ref_column)
    sample_rows: Optional[List[Dict]] = None


class SchemaManager:
    """
    Manages database schema extraction, caching, and formatting.

    This class provides methods to:
    - Extract schema from SQLite databases
    - Cache schema information for performance
    - Format schema for LLM prompts
    - Identify relevant tables for queries
    """

    def __init__(self, database_path: str, cache_enabled: bool = True):
        """
        Initialize schema manager.

        Args:
            database_path: Path to SQLite database
            cache_enabled: Whether to enable schema caching
        """
        self.database_path = database_path
        self.cache_enabled = cache_enabled
        self._schema_cache: Dict[str, TableInfo] = {}
        self._connection = None

        # Load schema on initialization
        self._load_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (create if needed)."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.database_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _load_schema(self):
        """Load complete schema from database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        tables = [row[0] for row in cursor.fetchall()]

        # Load info for each table
        for table_name in tables:
            self._schema_cache[table_name] = self._load_table_info(table_name)

    def _load_table_info(self, table_name: str) -> TableInfo:
        """
        Load information for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            TableInfo object with complete table information
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = []

        for row in cursor.fetchall():
            columns.append(ColumnInfo(
                name=row[1],
                type=row[2],
                nullable=not row[3],  # notnull flag
                primary_key=bool(row[5]),  # pk flag
                default_value=row[4]  # dflt_value
            ))

        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = []

        for row in cursor.fetchall():
            foreign_keys.append((
                row[3],  # from column
                row[2],  # to table
                row[4]   # to column
            ))

        # Get sample rows
        sample_rows = self._get_sample_rows(table_name, limit=3)

        return TableInfo(
            name=table_name,
            columns=columns,
            foreign_keys=foreign_keys,
            sample_rows=sample_rows
        )

    def _get_sample_rows(self, table_name: str, limit: int = 3) -> List[Dict]:
        """
        Get sample rows from table.

        Args:
            table_name: Name of the table
            limit: Number of sample rows to retrieve

        Returns:
            List of dictionaries representing rows
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []

    def get_all_tables(self) -> List[str]:
        """
        Get list of all table names.

        Returns:
            List of table names
        """
        return list(self._schema_cache.keys())

    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """
        Get information for specific table.

        Args:
            table_name: Name of the table

        Returns:
            TableInfo object or None if table doesn't exist
        """
        return self._schema_cache.get(table_name)

    def get_relevant_tables(
        self,
        question: str,
        max_tables: int = 10
    ) -> List[str]:
        """
        Identify tables relevant to a question using keyword matching.

        Args:
            question: Natural language question
            max_tables: Maximum number of tables to return

        Returns:
            List of relevant table names, sorted by relevance
        """
        question_lower = question.lower()

        # Extract potential keywords
        keywords = self._extract_keywords(question_lower)

        # Score each table
        table_scores = []

        for table_name, table_info in self._schema_cache.items():
            score = 0

            # Check table name
            table_lower = table_name.lower()
            for keyword in keywords:
                if keyword in table_lower:
                    score += 5

            # Check column names
            for column in table_info.columns:
                column_lower = column.name.lower()
                for keyword in keywords:
                    if keyword in column_lower:
                        score += 2

            # Check sample values
            if table_info.sample_rows:
                for row in table_info.sample_rows:
                    for value in row.values():
                        if value and str(value).lower() in question_lower:
                            score += 1

            if score > 0:
                table_scores.append((table_name, score))

        # Sort by score and return top N
        table_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_tables = [name for name, _ in table_scores[:max_tables]]

        # If no matches, return all tables (let LLM figure it out)
        if not relevant_tables:
            relevant_tables = list(self._schema_cache.keys())[:max_tables]

        # Add related tables via foreign keys
        relevant_tables = self._add_related_tables(relevant_tables)

        return relevant_tables[:max_tables]

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords from text.

        Args:
            text: Input text

        Returns:
            Set of keywords
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'are', 'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'show', 'get', 'give', 'list', 'find', 'tell', 'me', 'my'
        }

        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter stop words and short words
        keywords = {w for w in words if w not in stop_words and len(w) > 2}

        return keywords

    def _add_related_tables(self, table_names: List[str]) -> List[str]:
        """
        Add tables related via foreign keys.

        Args:
            table_names: Initial list of table names

        Returns:
            Extended list including related tables
        """
        related = set(table_names)

        for table_name in table_names:
            table_info = self.get_table_info(table_name)
            if table_info:
                # Add tables referenced by foreign keys
                for _, ref_table, _ in table_info.foreign_keys:
                    related.add(ref_table)

                # Add tables that reference this table
                for other_name, other_info in self._schema_cache.items():
                    for _, ref_table, _ in other_info.foreign_keys:
                        if ref_table == table_name:
                            related.add(other_name)

        return list(related)

    def format_schema_for_llm(
        self,
        table_names: Optional[List[str]] = None,
        include_samples: bool = True,
        format_style: str = "create_table"
    ) -> str:
        """
        Format schema for LLM consumption.

        Args:
            table_names: List of tables to include (None = all tables)
            include_samples: Whether to include sample rows
            format_style: Format style ('create_table', 'compact', 'json')

        Returns:
            Formatted schema string
        """
        if table_names is None:
            table_names = self.get_all_tables()

        if format_style == "create_table":
            return self._format_as_create_table(table_names, include_samples)
        elif format_style == "compact":
            return self._format_compact(table_names, include_samples)
        else:  # json
            return self._format_json(table_names, include_samples)

    def _format_as_create_table(
        self,
        table_names: List[str],
        include_samples: bool
    ) -> str:
        """Format schema as CREATE TABLE statements."""
        output = []

        for table_name in table_names:
            table_info = self.get_table_info(table_name)
            if not table_info:
                continue

            # CREATE TABLE statement
            lines = [f"CREATE TABLE {table_name} ("]

            # Columns
            column_defs = []
            for col in table_info.columns:
                col_def = f"  {col.name} {col.type}"
                if col.primary_key:
                    col_def += " PRIMARY KEY"
                if not col.nullable:
                    col_def += " NOT NULL"
                if col.default_value:
                    col_def += f" DEFAULT {col.default_value}"
                column_defs.append(col_def)

            lines.append(",\n".join(column_defs))
            lines.append(");")

            output.append("\n".join(lines))

            # Foreign keys
            if table_info.foreign_keys:
                fk_lines = []
                for from_col, to_table, to_col in table_info.foreign_keys:
                    fk_lines.append(
                        f"-- Foreign Key: {table_name}.{from_col} "
                        f"references {to_table}.{to_col}"
                    )
                output.append("\n".join(fk_lines))

            # Sample rows
            if include_samples and table_info.sample_rows:
                output.append(f"\n-- Sample rows from {table_name}:")
                for i, row in enumerate(table_info.sample_rows, 1):
                    row_str = ", ".join(f"{k}={v}" for k, v in row.items())
                    output.append(f"-- Row {i}: {row_str}")

            output.append("")  # Blank line

        return "\n".join(output)

    def _format_compact(
        self,
        table_names: List[str],
        include_samples: bool
    ) -> str:
        """Format schema in compact text format."""
        output = []

        for table_name in table_names:
            table_info = self.get_table_info(table_name)
            if not table_info:
                continue

            # Table name
            output.append(f"Table: {table_name}")

            # Columns
            col_strs = []
            for col in table_info.columns:
                col_str = f"{col.name} ({col.type}"
                if col.primary_key:
                    col_str += ", PK"
                if not col.nullable:
                    col_str += ", NOT NULL"
                col_str += ")"
                col_strs.append(col_str)

            output.append("Columns: " + ", ".join(col_strs))

            # Foreign keys
            if table_info.foreign_keys:
                fk_strs = [
                    f"{from_col}->{to_table}.{to_col}"
                    for from_col, to_table, to_col in table_info.foreign_keys
                ]
                output.append("Foreign Keys: " + ", ".join(fk_strs))

            output.append("")  # Blank line

        return "\n".join(output)

    def _format_json(
        self,
        table_names: List[str],
        include_samples: bool
    ) -> str:
        """Format schema as JSON."""
        import json

        schema_dict = {}

        for table_name in table_names:
            table_info = self.get_table_info(table_name)
            if not table_info:
                continue

            schema_dict[table_name] = {
                "columns": [
                    {
                        "name": col.name,
                        "type": col.type,
                        "nullable": col.nullable,
                        "primary_key": col.primary_key
                    }
                    for col in table_info.columns
                ],
                "foreign_keys": [
                    {
                        "from": from_col,
                        "to_table": to_table,
                        "to_column": to_col
                    }
                    for from_col, to_table, to_col in table_info.foreign_keys
                ]
            }

            if include_samples and table_info.sample_rows:
                schema_dict[table_name]["sample_rows"] = table_info.sample_rows

        return json.dumps(schema_dict, indent=2)

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
