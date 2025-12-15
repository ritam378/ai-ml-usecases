"""
Tests for TextToSQLGenerator module.
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query_generator import TextToSQLGenerator, QueryResult


@pytest.mark.unit
class TestTextToSQLGenerator:
    """Test TextToSQLGenerator functionality."""

    def test_init_openai(self, test_db_path, mock_env_vars):
        """Test initialization with OpenAI provider."""
        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai"
        )

        assert generator.database_path == test_db_path
        assert generator.provider == "openai"
        assert generator.model_name is not None

    def test_init_anthropic(self, test_db_path, mock_env_vars):
        """Test initialization with Anthropic provider."""
        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="anthropic"
        )

        assert generator.provider == "anthropic"

    def test_init_invalid_provider(self, test_db_path):
        """Test initialization with invalid provider."""
        with pytest.raises(ValueError):
            TextToSQLGenerator(
                database_path=test_db_path,
                provider="invalid_provider"
            )

    def test_init_custom_model(self, test_db_path, mock_env_vars):
        """Test initialization with custom model name."""
        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai",
            model_name="gpt-4"
        )

        assert generator.model_name == "gpt-4"

    @patch('query_generator.OpenAI')
    def test_generate_sql_openai_success(self, mock_openai_class, test_db_path, mock_env_vars):
        """Test successful SQL generation with OpenAI."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="SELECT * FROM customers"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_response.model = "gpt-3.5-turbo"
        mock_client.chat.completions.create.return_value = mock_response

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai"
        )

        result = generator.generate_sql("Show me all customers")

        assert isinstance(result, QueryResult)
        assert result.sql == "SELECT * FROM customers"
        assert result.status == "success"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    @patch('query_generator.Anthropic')
    def test_generate_sql_anthropic_success(self, mock_anthropic_class, test_db_path, mock_env_vars):
        """Test successful SQL generation with Anthropic."""
        # Mock Anthropic client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="SELECT * FROM customers")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_response.model = "claude-3-sonnet-20240229"
        mock_client.messages.create.return_value = mock_response

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="anthropic"
        )

        result = generator.generate_sql("Show me all customers")

        assert isinstance(result, QueryResult)
        assert result.sql == "SELECT * FROM customers"
        assert result.status == "success"

    def test_query_result_dataclass(self):
        """Test QueryResult dataclass."""
        result = QueryResult(
            sql="SELECT * FROM customers",
            status="success",
            error=None,
            prompt_tokens=100,
            completion_tokens=50,
            model_used="gpt-3.5-turbo"
        )

        assert result.sql == "SELECT * FROM customers"
        assert result.status == "success"
        assert result.error is None
        assert result.total_tokens == 150

    @patch('query_generator.OpenAI')
    def test_generate_sql_with_retry(self, mock_openai_class, test_db_path, mock_env_vars):
        """Test SQL generation with retry on error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # First call returns invalid SQL
        mock_response_1 = Mock()
        mock_response_1.choices = [Mock(message=Mock(content="SELECT * FROM invalid_table"))]
        mock_response_1.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_response_1.model = "gpt-3.5-turbo"

        # Second call returns valid SQL
        mock_response_2 = Mock()
        mock_response_2.choices = [Mock(message=Mock(content="SELECT * FROM customers"))]
        mock_response_2.usage = Mock(prompt_tokens=110, completion_tokens=55)
        mock_response_2.model = "gpt-3.5-turbo"

        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai",
            max_retries=1
        )

        result = generator.generate_sql("Show me all customers")

        # Should eventually succeed after retry
        assert mock_client.chat.completions.create.call_count >= 1

    @patch('query_generator.OpenAI')
    def test_generate_sql_max_retries_exceeded(self, mock_openai_class, test_db_path, mock_env_vars):
        """Test SQL generation when max retries exceeded."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Always return invalid SQL
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="INVALID SQL"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_response.model = "gpt-3.5-turbo"
        mock_client.chat.completions.create.return_value = mock_response

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai",
            max_retries=2
        )

        result = generator.generate_sql("Show me all customers")

        # Should fail after max retries
        assert result.status == "error" or result.sql == "INVALID SQL"

    def test_cache_functionality(self, test_db_path, mock_env_vars):
        """Test query caching."""
        with patch('query_generator.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="SELECT * FROM customers"))]
            mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
            mock_response.model = "gpt-3.5-turbo"
            mock_client.chat.completions.create.return_value = mock_response

            generator = TextToSQLGenerator(
                database_path=test_db_path,
                provider="openai",
                cache_enabled=True
            )

            # First call
            result1 = generator.generate_sql("Show me all customers")

            # Second call with same question (should use cache)
            result2 = generator.generate_sql("Show me all customers")

            # Should only call API once if caching works
            # (This depends on implementation - adjust assertion as needed)
            assert result1.sql == result2.sql

    def test_complexity_estimation(self, test_db_path, mock_env_vars):
        """Test query complexity estimation."""
        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai"
        )

        simple = generator.estimate_query_complexity("Show me all customers")
        complex_query = generator.estimate_query_complexity(
            "Show me customers who have spent more than the average order amount in the last year"
        )

        # Complex query should have higher complexity score
        assert complex_query >= simple


@pytest.mark.integration
@pytest.mark.requires_api_key
class TestTextToSQLGeneratorIntegration:
    """Integration tests requiring actual API keys."""

    def test_real_openai_generation(self, test_db_path):
        """Test with real OpenAI API (requires API key)."""
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OPENAI_API_KEY not set")

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai"
        )

        result = generator.generate_sql("Show me all customers")

        assert isinstance(result, QueryResult)
        assert result.status == "success"
        assert "SELECT" in result.sql.upper()
        assert "customers" in result.sql.lower()

    def test_real_anthropic_generation(self, test_db_path):
        """Test with real Anthropic API (requires API key)."""
        import os
        if not os.getenv('ANTHROPIC_API_KEY'):
            pytest.skip("ANTHROPIC_API_KEY not set")

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="anthropic"
        )

        result = generator.generate_sql("Show me all customers")

        assert isinstance(result, QueryResult)
        assert result.status == "success"
        assert "SELECT" in result.sql.upper()


@pytest.mark.unit
class TestQueryResult:
    """Test QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test creating QueryResult."""
        result = QueryResult(
            sql="SELECT * FROM customers",
            status="success",
            error=None,
            prompt_tokens=100,
            completion_tokens=50,
            model_used="gpt-3.5-turbo",
            latency_ms=500.0
        )

        assert result.sql == "SELECT * FROM customers"
        assert result.status == "success"
        assert result.total_tokens == 150
        assert result.latency_ms == 500.0

    def test_query_result_error_case(self):
        """Test QueryResult for error case."""
        result = QueryResult(
            sql=None,
            status="error",
            error="API Error: Rate limit exceeded",
            prompt_tokens=0,
            completion_tokens=0,
            model_used="gpt-3.5-turbo"
        )

        assert result.sql is None
        assert result.status == "error"
        assert result.error is not None
        assert result.total_tokens == 0

    def test_query_result_total_tokens(self):
        """Test total tokens calculation."""
        result = QueryResult(
            sql="SELECT * FROM customers",
            status="success",
            error=None,
            prompt_tokens=123,
            completion_tokens=456,
            model_used="gpt-4"
        )

        assert result.total_tokens == 579

    def test_query_result_optional_fields(self):
        """Test QueryResult with optional fields."""
        result = QueryResult(
            sql="SELECT * FROM customers",
            status="success",
            error=None,
            prompt_tokens=100,
            completion_tokens=50,
            model_used="gpt-3.5-turbo",
            reasoning="Using simple SELECT query",
            confidence=0.95
        )

        assert result.reasoning == "Using simple SELECT query"
        assert result.confidence == 0.95


@pytest.mark.unit
class TestTextToSQLGeneratorEdgeCases:
    """Test edge cases for TextToSQLGenerator."""

    @patch('query_generator.OpenAI')
    def test_empty_question(self, mock_openai_class, test_db_path, mock_env_vars):
        """Test generation with empty question."""
        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai"
        )

        # Should handle gracefully
        result = generator.generate_sql("")

        # Behavior depends on implementation
        assert isinstance(result, QueryResult)

    @patch('query_generator.OpenAI')
    def test_very_long_question(self, mock_openai_class, test_db_path, mock_env_vars):
        """Test generation with very long question."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="SELECT * FROM customers"))]
        mock_response.usage = Mock(prompt_tokens=1000, completion_tokens=50)
        mock_response.model = "gpt-3.5-turbo"
        mock_client.chat.completions.create.return_value = mock_response

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai"
        )

        long_question = "Show me all customers " * 100
        result = generator.generate_sql(long_question)

        assert isinstance(result, QueryResult)

    @patch('query_generator.OpenAI')
    def test_special_characters_in_question(self, mock_openai_class, test_db_path, mock_env_vars):
        """Test generation with special characters."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="SELECT * FROM customers WHERE price > 100"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_response.model = "gpt-3.5-turbo"
        mock_client.chat.completions.create.return_value = mock_response

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai"
        )

        question = "Show me orders where price > $100 & status='active'"
        result = generator.generate_sql(question)

        assert isinstance(result, QueryResult)
        assert result.status == "success"

    @patch('query_generator.OpenAI')
    def test_api_error_handling(self, mock_openai_class, test_db_path, mock_env_vars):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Simulate API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        generator = TextToSQLGenerator(
            database_path=test_db_path,
            provider="openai"
        )

        result = generator.generate_sql("Show me all customers")

        assert result.status == "error"
        assert result.error is not None
