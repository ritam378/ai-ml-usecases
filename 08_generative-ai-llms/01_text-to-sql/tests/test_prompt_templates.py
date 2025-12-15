"""
Tests for PromptTemplates module.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_templates import PromptTemplates


@pytest.mark.unit
class TestPromptTemplates:
    """Test PromptTemplates functionality."""

    def test_init(self):
        """Test PromptTemplates initialization."""
        templates = PromptTemplates()

        assert templates is not None
        assert hasattr(templates, 'few_shot_examples')

    def test_few_shot_examples_exist(self):
        """Test that few-shot examples are defined."""
        templates = PromptTemplates()

        assert isinstance(templates.few_shot_examples, list)
        assert len(templates.few_shot_examples) > 0

    def test_few_shot_example_structure(self):
        """Test structure of few-shot examples."""
        templates = PromptTemplates()

        for example in templates.few_shot_examples:
            assert 'question' in example
            assert 'sql' in example
            assert 'explanation' in example
            assert isinstance(example['question'], str)
            assert isinstance(example['sql'], str)
            assert isinstance(example['explanation'], str)

    def test_get_system_message(self):
        """Test getting system message."""
        templates = PromptTemplates()
        message = templates.get_system_message()

        assert isinstance(message, str)
        assert len(message) > 0
        assert 'SQL' in message or 'sql' in message

    def test_build_text_to_sql_prompt(self, sample_schema, sample_question):
        """Test building complete prompt."""
        templates = PromptTemplates()

        prompt = templates.build_text_to_sql_prompt(
            schema=sample_schema,
            question=sample_question,
            include_examples=True
        )

        assert isinstance(prompt, str)
        assert sample_schema in prompt
        assert sample_question in prompt
        assert len(prompt) > 0

    def test_build_prompt_without_examples(self, sample_schema, sample_question):
        """Test building prompt without few-shot examples."""
        templates = PromptTemplates()

        prompt = templates.build_text_to_sql_prompt(
            schema=sample_schema,
            question=sample_question,
            include_examples=False
        )

        assert isinstance(prompt, str)
        assert sample_schema in prompt
        assert sample_question in prompt

        # Should be shorter without examples
        prompt_with_examples = templates.build_text_to_sql_prompt(
            schema=sample_schema,
            question=sample_question,
            include_examples=True
        )
        assert len(prompt) < len(prompt_with_examples)

    def test_build_retry_prompt(self, sample_schema, sample_question, sample_sql):
        """Test building retry prompt after error."""
        templates = PromptTemplates()

        error_message = "Table 'customer' not found"
        retry_prompt = templates.build_retry_prompt(
            original_question=sample_question,
            failed_sql=sample_sql,
            error_message=error_message,
            schema=sample_schema
        )

        assert isinstance(retry_prompt, str)
        assert error_message in retry_prompt
        assert sample_sql in retry_prompt
        assert sample_question in retry_prompt

    def test_format_few_shot_examples(self):
        """Test formatting few-shot examples."""
        templates = PromptTemplates()

        formatted = templates.format_few_shot_examples(num_examples=3)

        assert isinstance(formatted, str)
        assert len(formatted) > 0

        # Should contain question, SQL, and explanation
        assert 'Question:' in formatted or 'question' in formatted.lower()
        assert 'SQL:' in formatted or 'SELECT' in formatted

    def test_format_few_shot_examples_limit(self):
        """Test limiting number of few-shot examples."""
        templates = PromptTemplates()

        formatted_3 = templates.format_few_shot_examples(num_examples=3)
        formatted_5 = templates.format_few_shot_examples(num_examples=5)

        # More examples should result in longer prompt
        assert len(formatted_5) > len(formatted_3)

    def test_get_example_by_type(self):
        """Test getting examples by type/category."""
        templates = PromptTemplates()

        # Should have different types of examples
        all_examples = templates.few_shot_examples
        assert len(all_examples) >= 3  # At least a few examples

        # Check for variety in examples
        questions = [ex['question'] for ex in all_examples]
        assert len(set(questions)) == len(questions)  # All unique

    def test_system_message_content(self):
        """Test system message contains key instructions."""
        templates = PromptTemplates()
        message = templates.get_system_message()

        # Should mention key concepts
        message_lower = message.lower()
        assert any(word in message_lower for word in ['sql', 'query', 'database'])

    def test_prompt_includes_schema_instructions(self, sample_schema, sample_question):
        """Test that prompt includes schema usage instructions."""
        templates = PromptTemplates()

        prompt = templates.build_text_to_sql_prompt(
            schema=sample_schema,
            question=sample_question,
            include_examples=True
        )

        # Should mention schema or tables
        prompt_lower = prompt.lower()
        assert 'schema' in prompt_lower or 'table' in prompt_lower

    def test_retry_prompt_emphasizes_error(self, sample_schema, sample_question, sample_sql):
        """Test that retry prompt emphasizes the error."""
        templates = PromptTemplates()

        error_message = "Syntax error near 'FROM'"
        retry_prompt = templates.build_retry_prompt(
            original_question=sample_question,
            failed_sql=sample_sql,
            error_message=error_message,
            schema=sample_schema
        )

        # Error should be prominently featured
        retry_lower = retry_prompt.lower()
        assert 'error' in retry_lower or 'failed' in retry_lower
        assert error_message in retry_prompt


@pytest.mark.unit
class TestPromptTemplatesEdgeCases:
    """Test edge cases for PromptTemplates."""

    def test_empty_schema(self, sample_question):
        """Test building prompt with empty schema."""
        templates = PromptTemplates()

        prompt = templates.build_text_to_sql_prompt(
            schema="",
            question=sample_question,
            include_examples=False
        )

        # Should still create a prompt
        assert isinstance(prompt, str)
        assert sample_question in prompt

    def test_long_question(self, sample_schema):
        """Test building prompt with very long question."""
        templates = PromptTemplates()

        long_question = "Show me all customers " * 50  # Very long
        prompt = templates.build_text_to_sql_prompt(
            schema=sample_schema,
            question=long_question,
            include_examples=False
        )

        assert isinstance(prompt, str)
        assert long_question in prompt

    def test_special_characters_in_question(self, sample_schema):
        """Test handling special characters in question."""
        templates = PromptTemplates()

        special_question = "Show me customer's orders where price > $100 & status='active'"
        prompt = templates.build_text_to_sql_prompt(
            schema=sample_schema,
            question=special_question,
            include_examples=False
        )

        assert isinstance(prompt, str)
        assert special_question in prompt

    def test_zero_examples(self):
        """Test formatting with zero examples."""
        templates = PromptTemplates()

        formatted = templates.format_few_shot_examples(num_examples=0)

        # Should return empty or minimal string
        assert isinstance(formatted, str)
        assert len(formatted) == 0 or formatted.isspace()

    def test_more_examples_than_available(self):
        """Test requesting more examples than available."""
        templates = PromptTemplates()

        total_examples = len(templates.few_shot_examples)
        formatted = templates.format_few_shot_examples(num_examples=total_examples + 10)

        # Should use all available examples
        assert isinstance(formatted, str)
        assert len(formatted) > 0


@pytest.mark.integration
class TestPromptTemplatesIntegration:
    """Integration tests for PromptTemplates."""

    def test_full_prompt_generation_workflow(self, sample_schema, sample_question):
        """Test complete prompt generation workflow."""
        templates = PromptTemplates()

        # 1. Get system message
        system_msg = templates.get_system_message()
        assert len(system_msg) > 0

        # 2. Build initial prompt
        prompt = templates.build_text_to_sql_prompt(
            schema=sample_schema,
            question=sample_question,
            include_examples=True
        )
        assert len(prompt) > 0

        # 3. Simulate error and build retry prompt
        retry_prompt = templates.build_retry_prompt(
            original_question=sample_question,
            failed_sql="SELECT * FROM customer",  # Wrong table name
            error_message="Table 'customer' does not exist",
            schema=sample_schema
        )
        assert len(retry_prompt) > 0
        assert "error" in retry_prompt.lower()

    def test_prompt_quality_for_different_complexities(self):
        """Test prompt generation for queries of different complexity."""
        templates = PromptTemplates()

        schema = """
        CREATE TABLE customers (id INTEGER, name TEXT);
        CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL);
        """

        queries = [
            "Show all customers",  # Simple
            "What is the average order amount?",  # Medium
            "Show customers who spent more than the average",  # Complex
        ]

        for question in queries:
            prompt = templates.build_text_to_sql_prompt(
                schema=schema,
                question=question,
                include_examples=True
            )

            # All should produce valid prompts
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Reasonable minimum length
            assert question in prompt
            assert schema in prompt
