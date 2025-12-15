"""
Tests for QueryValidator module.
"""

import pytest
import sqlite3
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query_validator import QueryValidator


@pytest.mark.unit
class TestQueryValidator:
    """Test QueryValidator functionality."""

    def test_init_read_only(self, test_db_path):
        """Test QueryValidator initialization in read-only mode."""
        validator = QueryValidator(test_db_path, read_only=True)

        assert validator.database_path == test_db_path
        assert validator.read_only is True

    def test_init_writable(self, test_db_path):
        """Test QueryValidator initialization in writable mode."""
        validator = QueryValidator(test_db_path, read_only=False)

        assert validator.read_only is False

    def test_validate_query_valid_select(self, test_db_path):
        """Test validation of valid SELECT query."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM customers"

        result = validator.validate_query(sql)

        assert result['is_valid'] is True
        assert result['is_safe'] is True
        assert result['error'] is None

    def test_validate_query_empty(self, test_db_path):
        """Test validation of empty query."""
        validator = QueryValidator(test_db_path)

        result = validator.validate_query("")

        assert result['is_valid'] is False
        assert 'empty' in result['error'].lower()

    def test_validate_query_dangerous_drop(self, test_db_path):
        """Test validation rejects DROP statement."""
        validator = QueryValidator(test_db_path)
        sql = "DROP TABLE customers"

        result = validator.validate_query(sql)

        assert result['is_valid'] is False
        assert result['is_safe'] is False
        assert 'dangerous' in result['error'].lower() or 'not allowed' in result['error'].lower()

    def test_validate_query_dangerous_delete(self, test_db_path):
        """Test validation rejects DELETE statement."""
        validator = QueryValidator(test_db_path)
        sql = "DELETE FROM customers WHERE customer_id = 1"

        result = validator.validate_query(sql)

        assert result['is_valid'] is False
        assert result['is_safe'] is False

    def test_validate_query_dangerous_update(self, test_db_path):
        """Test validation rejects UPDATE statement."""
        validator = QueryValidator(test_db_path)
        sql = "UPDATE customers SET name = 'Test' WHERE customer_id = 1"

        result = validator.validate_query(sql)

        assert result['is_valid'] is False
        assert result['is_safe'] is False

    def test_validate_query_dangerous_insert(self, test_db_path):
        """Test validation rejects INSERT statement."""
        validator = QueryValidator(test_db_path)
        sql = "INSERT INTO customers (name, email) VALUES ('Test', 'test@test.com')"

        result = validator.validate_query(sql)

        assert result['is_valid'] is False
        assert result['is_safe'] is False

    def test_validate_query_nonexistent_table(self, test_db_path):
        """Test validation of query with non-existent table."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM nonexistent_table"

        result = validator.validate_query(sql)

        assert result['is_valid'] is False
        assert result['tables_exist'] is False

    def test_validate_query_syntax_error(self, test_db_path):
        """Test validation of query with syntax error."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FORM customers"  # FORM instead of FROM

        result = validator.validate_query(sql)

        assert result['is_valid'] is False

    def test_validate_query_with_join(self, test_db_path):
        """Test validation of valid JOIN query."""
        validator = QueryValidator(test_db_path)
        sql = """
            SELECT c.*, o.*
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
        """

        result = validator.validate_query(sql)

        assert result['is_valid'] is True
        assert result['is_safe'] is True

    def test_validate_query_with_where(self, test_db_path):
        """Test validation of query with WHERE clause."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM customers WHERE city = 'New York'"

        result = validator.validate_query(sql)

        assert result['is_valid'] is True

    def test_validate_query_with_aggregation(self, test_db_path):
        """Test validation of query with aggregation."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT COUNT(*) as total FROM customers"

        result = validator.validate_query(sql)

        assert result['is_valid'] is True

    def test_validate_query_with_group_by(self, test_db_path):
        """Test validation of query with GROUP BY."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT city, COUNT(*) FROM customers GROUP BY city"

        result = validator.validate_query(sql)

        assert result['is_valid'] is True

    def test_validate_query_with_order_by(self, test_db_path):
        """Test validation of query with ORDER BY."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM customers ORDER BY name"

        result = validator.validate_query(sql)

        assert result['is_valid'] is True

    def test_validate_query_with_limit(self, test_db_path):
        """Test validation of query with LIMIT."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM customers LIMIT 10"

        result = validator.validate_query(sql)

        assert result['is_valid'] is True

    def test_execute_query_valid(self, test_db_path):
        """Test executing valid query."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM customers"

        result = validator.execute_query(sql)

        assert result['success'] is True
        assert result['rows'] is not None
        assert isinstance(result['rows'], list)
        assert result['error'] is None

    def test_execute_query_invalid(self, test_db_path):
        """Test executing invalid query."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM nonexistent"

        result = validator.execute_query(sql)

        assert result['success'] is False
        assert result['error'] is not None

    def test_execute_query_dangerous(self, test_db_path):
        """Test executing dangerous query."""
        validator = QueryValidator(test_db_path)
        sql = "DROP TABLE customers"

        result = validator.execute_query(sql)

        assert result['success'] is False
        assert 'dangerous' in result['error'].lower() or 'not allowed' in result['error'].lower()

    def test_execute_query_returns_columns(self, test_db_path):
        """Test that execute_query returns column names."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT customer_id, name FROM customers LIMIT 1"

        result = validator.execute_query(sql)

        assert result['success'] is True
        assert 'columns' in result
        assert 'customer_id' in result['columns']
        assert 'name' in result['columns']

    def test_execute_query_dry_run(self, test_db_path):
        """Test executing query in dry run mode."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM customers"

        result = validator.execute_query(sql, dry_run=True)

        assert result['success'] is True
        # In dry run, might not return actual rows

    def test_get_query_plan(self, test_db_path):
        """Test getting query execution plan."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM customers WHERE city = 'New York'"

        plan = validator.get_query_plan(sql)

        assert plan is not None
        assert isinstance(plan, str)
        assert 'SCAN' in plan or 'SEARCH' in plan or 'explain' in plan.lower()

    def test_check_table_exists_valid(self, test_db_path):
        """Test checking existence of valid table."""
        validator = QueryValidator(test_db_path)

        exists = validator.check_table_exists("customers")

        assert exists is True

    def test_check_table_exists_invalid(self, test_db_path):
        """Test checking existence of invalid table."""
        validator = QueryValidator(test_db_path)

        exists = validator.check_table_exists("nonexistent")

        assert exists is False

    def test_extract_table_names(self, test_db_path):
        """Test extracting table names from query."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT * FROM customers JOIN orders ON customers.customer_id = orders.customer_id"

        tables = validator.extract_table_names(sql)

        assert isinstance(tables, list)
        assert "customers" in tables
        assert "orders" in tables

    def test_extract_table_names_with_alias(self, test_db_path):
        """Test extracting table names from query with aliases."""
        validator = QueryValidator(test_db_path)
        sql = "SELECT c.* FROM customers c"

        tables = validator.extract_table_names(sql)

        assert "customers" in tables

    def test_is_select_query_valid(self, test_db_path):
        """Test detecting SELECT query."""
        validator = QueryValidator(test_db_path)

        assert validator.is_select_query("SELECT * FROM customers") is True
        assert validator.is_select_query("  select * from customers  ") is True

    def test_is_select_query_invalid(self, test_db_path):
        """Test detecting non-SELECT query."""
        validator = QueryValidator(test_db_path)

        assert validator.is_select_query("DELETE FROM customers") is False
        assert validator.is_select_query("UPDATE customers SET name='Test'") is False
        assert validator.is_select_query("INSERT INTO customers VALUES (1, 'Test')") is False


@pytest.mark.integration
class TestQueryValidatorIntegration:
    """Integration tests for QueryValidator."""

    def test_full_validation_workflow(self, test_db_path):
        """Test complete validation and execution workflow."""
        validator = QueryValidator(test_db_path)

        # 1. Validate query
        sql = "SELECT * FROM customers WHERE city = 'New York'"
        validation = validator.validate_query(sql)

        assert validation['is_valid'] is True

        # 2. Execute query
        execution = validator.execute_query(sql)

        assert execution['success'] is True
        assert execution['rows'] is not None

    def test_validation_prevents_dangerous_execution(self, test_db_path):
        """Test that dangerous queries are blocked."""
        validator = QueryValidator(test_db_path)

        dangerous_queries = [
            "DROP TABLE customers",
            "DELETE FROM customers",
            "UPDATE customers SET name = 'Hacked'",
            "INSERT INTO customers VALUES (999, 'Test', 'test@test.com', NULL)"
        ]

        for sql in dangerous_queries:
            validation = validator.validate_query(sql)
            assert validation['is_safe'] is False

            execution = validator.execute_query(sql)
            assert execution['success'] is False

    def test_complex_query_validation_and_execution(self, test_db_path):
        """Test validation and execution of complex query."""
        validator = QueryValidator(test_db_path)

        sql = """
            SELECT
                c.name,
                c.city,
                COUNT(o.order_id) as order_count,
                SUM(o.total_amount) as total_spent
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.name, c.city
            ORDER BY total_spent DESC
            LIMIT 10
        """

        # Validate
        validation = validator.validate_query(sql)
        assert validation['is_valid'] is True

        # Execute
        execution = validator.execute_query(sql)
        assert execution['success'] is True


@pytest.mark.unit
class TestQueryValidatorEdgeCases:
    """Test edge cases for QueryValidator."""

    def test_multiline_query(self, test_db_path):
        """Test validation of multi-line query."""
        validator = QueryValidator(test_db_path)
        sql = """
            SELECT
                customer_id,
                name,
                email
            FROM customers
            WHERE city = 'New York'
        """

        result = validator.validate_query(sql)
        assert result['is_valid'] is True

    def test_query_with_comments(self, test_db_path):
        """Test validation of query with comments."""
        validator = QueryValidator(test_db_path)
        sql = """
            -- Get all customers from New York
            SELECT * FROM customers
            WHERE city = 'New York'  -- Filter by city
        """

        result = validator.validate_query(sql)
        assert result['is_valid'] is True

    def test_query_with_subquery(self, test_db_path):
        """Test validation of query with subquery."""
        validator = QueryValidator(test_db_path)
        sql = """
            SELECT * FROM customers
            WHERE customer_id IN (
                SELECT customer_id FROM orders WHERE total_amount > 100
            )
        """

        result = validator.validate_query(sql)
        assert result['is_valid'] is True

    def test_whitespace_only_query(self, test_db_path):
        """Test validation of whitespace-only query."""
        validator = QueryValidator(test_db_path)

        result = validator.validate_query("   \n\t   ")
        assert result['is_valid'] is False

    def test_case_insensitive_keywords(self, test_db_path):
        """Test that SQL keywords are case-insensitive."""
        validator = QueryValidator(test_db_path)

        # All should be valid
        queries = [
            "SELECT * FROM customers",
            "select * from customers",
            "SeLeCt * FrOm customers"
        ]

        for sql in queries:
            result = validator.validate_query(sql)
            assert result['is_valid'] is True

    def test_dangerous_keyword_detection_case_insensitive(self, test_db_path):
        """Test dangerous keyword detection is case-insensitive."""
        validator = QueryValidator(test_db_path)

        dangerous = [
            "DROP TABLE customers",
            "drop table customers",
            "DrOp TaBlE customers"
        ]

        for sql in dangerous:
            result = validator.validate_query(sql)
            assert result['is_safe'] is False
