"""
Pytest configuration and shared fixtures for Text-to-SQL tests.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def test_db_path() -> Generator[str, None, None]:
    """
    Create a temporary test database for the session.

    Returns:
        Path to temporary test database
    """
    # Use the existing sample database
    current_dir = Path(__file__).parent
    db_path = current_dir.parent / "data" / "sample_database.db"

    if db_path.exists():
        yield str(db_path)
    else:
        # Create a minimal test database if sample doesn't exist
        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            temp_db_path = f.name

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Create minimal schema
        cursor.execute("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                city TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                category TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                total_amount REAL,
                status TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """)

        # Insert test data
        cursor.execute(
            "INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com', 'New York')"
        )
        cursor.execute(
            "INSERT INTO customers VALUES (2, 'Bob', 'bob@example.com', 'Los Angeles')"
        )
        cursor.execute(
            "INSERT INTO products VALUES (1, 'Laptop', 999.99, 'Electronics')"
        )
        cursor.execute(
            "INSERT INTO products VALUES (2, 'Mouse', 29.99, 'Electronics')"
        )
        cursor.execute(
            "INSERT INTO orders VALUES (1, 1, 1029.98, 'completed')"
        )

        conn.commit()
        conn.close()

        yield temp_db_path

        # Cleanup
        try:
            os.unlink(temp_db_path)
        except:
            pass


@pytest.fixture
def db_connection(test_db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """
    Provide a database connection for tests.

    Args:
        test_db_path: Path to test database

    Yields:
        SQLite connection
    """
    conn = sqlite3.connect(test_db_path)
    yield conn
    conn.close()


@pytest.fixture
def sample_schema() -> str:
    """
    Provide a sample database schema string.

    Returns:
        Schema string in CREATE TABLE format
    """
    return """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    city TEXT
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    total_amount REAL,
    status TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
""".strip()


@pytest.fixture
def sample_question() -> str:
    """Provide a sample natural language question."""
    return "Show me all customers from New York"


@pytest.fixture
def sample_sql() -> str:
    """Provide a sample SQL query."""
    return "SELECT * FROM customers WHERE city = 'New York'"


@pytest.fixture
def mock_openai_response():
    """
    Mock OpenAI API response.

    Returns:
        Mock response object
    """
    class MockChoice:
        def __init__(self, content: str):
            self.message = type('Message', (), {'content': content})

    class MockUsage:
        def __init__(self):
            self.prompt_tokens = 100
            self.completion_tokens = 50
            self.total_tokens = 150

    class MockResponse:
        def __init__(self, sql: str):
            self.choices = [MockChoice(sql)]
            self.usage = MockUsage()
            self.model = "gpt-3.5-turbo"

    return MockResponse


@pytest.fixture
def mock_anthropic_response():
    """
    Mock Anthropic API response.

    Returns:
        Mock response object
    """
    class MockUsage:
        def __init__(self):
            self.input_tokens = 100
            self.output_tokens = 50

    class MockContent:
        def __init__(self, text: str):
            self.text = text

    class MockResponse:
        def __init__(self, sql: str):
            self.content = [MockContent(sql)]
            self.usage = MockUsage()
            self.model = "claude-3-sonnet-20240229"

    return MockResponse


@pytest.fixture
def test_queries():
    """
    Provide sample test queries with expected SQL.

    Returns:
        List of test query dicts
    """
    return [
        {
            "question": "Show me all customers",
            "expected_sql": "SELECT * FROM customers",
            "complexity": "simple",
            "category": "basic_query"
        },
        {
            "question": "What is the total amount of all orders?",
            "expected_sql": "SELECT SUM(total_amount) FROM orders",
            "complexity": "simple",
            "category": "aggregation"
        },
        {
            "question": "Show me customers and their orders",
            "expected_sql": """
                SELECT c.*, o.*
                FROM customers c
                LEFT JOIN orders o ON c.customer_id = o.customer_id
            """,
            "complexity": "medium",
            "category": "joins"
        }
    ]


# Environment variable fixtures
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-456")


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API key"
    )
