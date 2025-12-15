"""
Tests for SchemaManager module.
"""

import pytest
import sqlite3
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schema_manager import SchemaManager, TableInfo, ColumnInfo


@pytest.mark.unit
class TestSchemaManager:
    """Test SchemaManager functionality."""

    def test_init(self, test_db_path):
        """Test SchemaManager initialization."""
        manager = SchemaManager(test_db_path)

        assert manager.database_path == test_db_path
        assert manager.cache_enabled is True
        assert len(manager._schema_cache) > 0

    def test_init_with_cache_disabled(self, test_db_path):
        """Test initialization with caching disabled."""
        manager = SchemaManager(test_db_path, cache_enabled=False)

        assert manager.cache_enabled is False

    def test_get_schema(self, test_db_path):
        """Test getting complete schema."""
        manager = SchemaManager(test_db_path)
        schema = manager.get_schema()

        assert hasattr(schema, 'tables')
        assert len(schema.tables) > 0
        assert all(isinstance(table, TableInfo) for table in schema.tables)

    def test_get_table_schema(self, test_db_path):
        """Test getting single table schema."""
        manager = SchemaManager(test_db_path)
        table_schema = manager.get_table_schema("customers")

        assert isinstance(table_schema, TableInfo)
        assert table_schema.name == "customers"
        assert len(table_schema.columns) > 0

    def test_get_table_schema_nonexistent(self, test_db_path):
        """Test getting schema for non-existent table."""
        manager = SchemaManager(test_db_path)

        with pytest.raises(KeyError):
            manager.get_table_schema("nonexistent_table")

    def test_column_info_structure(self, test_db_path):
        """Test ColumnInfo structure."""
        manager = SchemaManager(test_db_path)
        table = manager.get_table_schema("customers")

        for column in table.columns:
            assert isinstance(column, ColumnInfo)
            assert hasattr(column, 'name')
            assert hasattr(column, 'type')
            assert hasattr(column, 'nullable')
            assert hasattr(column, 'primary_key')

    def test_primary_key_detection(self, test_db_path):
        """Test primary key detection."""
        manager = SchemaManager(test_db_path)
        table = manager.get_table_schema("customers")

        # Find primary key column
        pk_columns = [col for col in table.columns if col.primary_key]
        assert len(pk_columns) == 1
        assert pk_columns[0].name == "customer_id"

    def test_foreign_key_detection(self, test_db_path):
        """Test foreign key detection."""
        manager = SchemaManager(test_db_path)

        # Orders table should have foreign key to customers
        orders_table = manager.get_table_schema("orders")
        assert len(orders_table.foreign_keys) > 0

        # Check foreign key structure
        fk = orders_table.foreign_keys[0]
        assert len(fk) == 3  # (column, ref_table, ref_column)
        assert fk[0] == "customer_id"  # Column name
        assert fk[1] == "customers"  # Referenced table

    def test_format_schema_create_table(self, test_db_path):
        """Test CREATE TABLE schema formatting."""
        manager = SchemaManager(test_db_path)
        schema_str = manager.format_schema(format_type="create_table")

        assert "CREATE TABLE" in schema_str
        assert "customers" in schema_str
        assert "PRIMARY KEY" in schema_str

    def test_format_schema_compact(self, test_db_path):
        """Test compact schema formatting."""
        manager = SchemaManager(test_db_path)
        schema_str = manager.format_schema(format_type="compact")

        assert "customers" in schema_str
        assert len(schema_str) < len(manager.format_schema(format_type="create_table"))

    def test_format_schema_json(self, test_db_path):
        """Test JSON schema formatting."""
        manager = SchemaManager(test_db_path)
        schema_str = manager.format_schema(format_type="json")

        assert "{" in schema_str and "}" in schema_str
        assert "customers" in schema_str
        assert "columns" in schema_str

    def test_format_schema_with_samples(self, test_db_path):
        """Test schema formatting with sample rows."""
        manager = SchemaManager(test_db_path)
        schema_str = manager.format_schema(
            format_type="create_table",
            include_sample_rows=True,
            num_sample_rows=2
        )

        assert "Sample rows:" in schema_str or "sample" in schema_str.lower()

    def test_identify_relevant_tables_simple(self, test_db_path):
        """Test relevant table identification for simple queries."""
        manager = SchemaManager(test_db_path)

        question = "Show me all customers"
        relevant = manager.identify_relevant_tables(question)

        assert "customers" in relevant
        assert isinstance(relevant, list)

    def test_identify_relevant_tables_multiple(self, test_db_path):
        """Test relevant table identification for multi-table queries."""
        manager = SchemaManager(test_db_path)

        question = "Show me customers and their orders"
        relevant = manager.identify_relevant_tables(question)

        assert "customers" in relevant
        assert "orders" in relevant

    def test_identify_relevant_tables_empty(self, test_db_path):
        """Test relevant table identification with unrelated question."""
        manager = SchemaManager(test_db_path)

        question = "What is the weather today?"
        relevant = manager.identify_relevant_tables(question)

        # Should return empty list or all tables as fallback
        assert isinstance(relevant, list)

    def test_get_sample_rows(self, test_db_path):
        """Test getting sample rows for a table."""
        manager = SchemaManager(test_db_path)
        samples = manager.get_sample_rows("customers", limit=2)

        assert isinstance(samples, list)
        assert len(samples) <= 2
        if len(samples) > 0:
            assert isinstance(samples[0], dict)

    def test_get_sample_rows_empty_table(self, test_db_path):
        """Test getting sample rows from empty table."""
        manager = SchemaManager(test_db_path)

        # Create temporary empty table
        conn = manager._get_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE TEMP TABLE empty_table (id INTEGER)")
        conn.commit()

        samples = manager.get_sample_rows("empty_table")
        assert samples == []

    def test_cache_usage(self, test_db_path):
        """Test that caching is working."""
        manager = SchemaManager(test_db_path, cache_enabled=True)

        # First call should populate cache
        schema1 = manager.get_table_schema("customers")

        # Second call should use cache
        schema2 = manager.get_table_schema("customers")

        # Should return same object (identity check)
        assert schema1 is schema2

    def test_table_exists(self, test_db_path):
        """Test table existence checking."""
        manager = SchemaManager(test_db_path)

        assert manager.table_exists("customers") is True
        assert manager.table_exists("nonexistent") is False

    def test_get_table_names(self, test_db_path):
        """Test getting all table names."""
        manager = SchemaManager(test_db_path)
        tables = manager.get_table_names()

        assert isinstance(tables, list)
        assert "customers" in tables
        assert "orders" in tables
        assert len(tables) >= 2

    def test_format_table_for_llm(self, test_db_path):
        """Test LLM-optimized table formatting."""
        manager = SchemaManager(test_db_path)
        formatted = manager.format_table_for_llm("customers", include_samples=False)

        assert "customers" in formatted
        assert "customer_id" in formatted
        assert isinstance(formatted, str)

    def test_get_column_names(self, test_db_path):
        """Test getting column names for a table."""
        manager = SchemaManager(test_db_path)
        columns = manager.get_column_names("customers")

        assert isinstance(columns, list)
        assert "customer_id" in columns
        assert "name" in columns

    def test_connection_reuse(self, test_db_path):
        """Test that database connection is reused."""
        manager = SchemaManager(test_db_path)

        conn1 = manager._get_connection()
        conn2 = manager._get_connection()

        # Should be same connection object
        assert conn1 is conn2


@pytest.mark.integration
class TestSchemaManagerIntegration:
    """Integration tests for SchemaManager."""

    def test_full_schema_extraction_workflow(self, test_db_path):
        """Test complete schema extraction workflow."""
        manager = SchemaManager(test_db_path)

        # Get schema
        schema = manager.get_schema()

        # Verify structure
        assert len(schema.tables) > 0

        # Format for LLM
        llm_format = manager.format_schema(format_type="create_table")
        assert len(llm_format) > 0

        # Identify relevant tables
        relevant = manager.identify_relevant_tables("customer orders")
        assert len(relevant) > 0

    def test_schema_with_real_database(self, test_db_path):
        """Test with actual sample database if available."""
        # Check if sample database exists
        sample_db = Path(test_db_path).parent.parent / "data" / "sample_database.db"

        if not sample_db.exists():
            pytest.skip("Sample database not found")

        manager = SchemaManager(str(sample_db))
        schema = manager.get_schema()

        # Verify expected tables
        table_names = [t.name for t in schema.tables]
        expected_tables = {"customers", "products", "orders", "order_items", "reviews"}

        assert expected_tables.issubset(set(table_names))


@pytest.mark.unit
class TestTableInfo:
    """Test TableInfo dataclass."""

    def test_table_info_creation(self):
        """Test TableInfo creation."""
        columns = [
            ColumnInfo("id", "INTEGER", False, True),
            ColumnInfo("name", "TEXT", False, False)
        ]

        table = TableInfo("test_table", columns, [])

        assert table.name == "test_table"
        assert len(table.columns) == 2
        assert table.foreign_keys == []

    def test_table_info_with_foreign_keys(self):
        """Test TableInfo with foreign keys."""
        columns = [ColumnInfo("id", "INTEGER", False, True)]
        fks = [("customer_id", "customers", "id")]

        table = TableInfo("orders", columns, fks)

        assert len(table.foreign_keys) == 1
        assert table.foreign_keys[0][0] == "customer_id"


@pytest.mark.unit
class TestColumnInfo:
    """Test ColumnInfo dataclass."""

    def test_column_info_creation(self):
        """Test ColumnInfo creation."""
        col = ColumnInfo("test_col", "TEXT", True, False, "default_value")

        assert col.name == "test_col"
        assert col.type == "TEXT"
        assert col.nullable is True
        assert col.primary_key is False
        assert col.default_value == "default_value"

    def test_column_info_minimal(self):
        """Test ColumnInfo with minimal args."""
        col = ColumnInfo("id", "INTEGER", False, True)

        assert col.name == "id"
        assert col.default_value is None
