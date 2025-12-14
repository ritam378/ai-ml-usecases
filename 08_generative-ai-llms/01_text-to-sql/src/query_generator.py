"""
Text-to-SQL Query Generator

This module provides the main interface for generating SQL queries from natural language.
"""

from typing import Dict, List, Optional, Tuple
import os
import time
from dataclasses import dataclass
import hashlib
from functools import lru_cache

try:
    import openai
except ImportError:
    openai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from .schema_manager import SchemaManager
from .prompt_templates import PromptTemplates
from .query_validator import QueryValidator


@dataclass
class QueryResult:
    """Result of SQL generation."""
    sql: str
    explanation: str
    confidence: float
    tokens_used: int
    latency: float
    model: str
    assumptions: List[str] = None

    def __post_init__(self):
        if self.assumptions is None:
            self.assumptions = []


class TextToSQLGenerator:
    """
    Main class for generating SQL queries from natural language questions.

    This class orchestrates schema management, prompt engineering, LLM interaction,
    and query validation to produce executable SQL queries.

    Attributes:
        schema_manager: Manages database schema extraction and formatting
        validator: Validates generated SQL queries
        cache: In-memory cache for generated queries
        provider: LLM provider ('openai' or 'anthropic')
    """

    def __init__(
        self,
        database_path: str,
        provider: str = "openai",
        model: Optional[str] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the Text-to-SQL generator.

        Args:
            database_path: Path to SQLite database
            provider: LLM provider ('openai' or 'anthropic')
            model: Specific model to use (e.g., 'gpt-4', 'claude-3-opus')
            cache_enabled: Whether to enable query caching
        """
        self.schema_manager = SchemaManager(database_path)
        self.validator = QueryValidator(database_path)
        self.provider = provider.lower()
        self.cache = {} if cache_enabled else None

        # Set default models
        if model is None:
            self.model = "gpt-4" if self.provider == "openai" else "claude-3-opus-20240229"
        else:
            self.model = model

        # Initialize API clients
        if self.provider == "openai":
            if openai is None:
                raise ImportError("openai package not installed. Run: pip install openai")
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_sql(
        self,
        question: str,
        max_retries: int = 2,
        include_explanation: bool = True
    ) -> QueryResult:
        """
        Generate SQL query from natural language question.

        Args:
            question: Natural language question
            max_retries: Maximum number of retry attempts on errors
            include_explanation: Whether to include natural language explanation

        Returns:
            QueryResult object containing SQL and metadata

        Raises:
            ValueError: If question is empty or invalid
            RuntimeError: If SQL generation fails after all retries
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Check cache first
        if self.cache is not None:
            cache_key = self._get_cache_key(question)
            if cache_key in self.cache:
                return self.cache[cache_key]

        # Determine query complexity and select model
        complexity = self._classify_complexity(question)
        model = self._select_model(complexity)

        # Generate SQL with retries
        result = self._generate_with_retry(
            question=question,
            max_retries=max_retries,
            model=model,
            include_explanation=include_explanation
        )

        # Cache result
        if self.cache is not None:
            cache_key = self._get_cache_key(question)
            self.cache[cache_key] = result

        return result

    def _generate_with_retry(
        self,
        question: str,
        max_retries: int,
        model: str,
        include_explanation: bool
    ) -> QueryResult:
        """
        Generate SQL with automatic retry on errors.

        Args:
            question: Natural language question
            max_retries: Maximum retry attempts
            model: Model to use for generation
            include_explanation: Whether to include explanation

        Returns:
            QueryResult object

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Add error feedback to question if retrying
                modified_question = question
                if attempt > 0 and last_error:
                    modified_question = (
                        f"{question}\n\n"
                        f"Previous attempt failed with error: {last_error}\n"
                        f"Please fix the SQL query."
                    )

                # Generate SQL
                result = self._generate_once(
                    question=modified_question,
                    model=model,
                    include_explanation=include_explanation
                )

                # Validate SQL
                is_valid, error_msg = self.validator.validate(result.sql)

                if is_valid:
                    # Dry run to ensure executability
                    can_execute, exec_error = self.validator.dry_run(result.sql)
                    if can_execute:
                        return result
                    else:
                        last_error = exec_error
                else:
                    last_error = error_msg

            except Exception as e:
                last_error = str(e)

        # All retries failed
        raise RuntimeError(
            f"Failed to generate valid SQL after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def _generate_once(
        self,
        question: str,
        model: str,
        include_explanation: bool
    ) -> QueryResult:
        """
        Generate SQL query once (no retry logic).

        Args:
            question: Natural language question
            model: Model to use
            include_explanation: Whether to include explanation

        Returns:
            QueryResult object
        """
        start_time = time.time()

        # 1. Identify relevant tables
        relevant_tables = self.schema_manager.get_relevant_tables(question)

        # 2. Get schema for those tables
        schema = self.schema_manager.format_schema_for_llm(relevant_tables)

        # 3. Build prompt
        prompt = PromptTemplates.build_prompt(
            schema=schema,
            question=question,
            include_explanation=include_explanation
        )

        # 4. Call LLM
        if self.provider == "openai":
            response = self._call_openai(prompt, model)
        else:  # anthropic
            response = self._call_anthropic(prompt, model)

        # 5. Parse response
        sql, explanation, assumptions = self._parse_response(response['content'])

        latency = time.time() - start_time

        return QueryResult(
            sql=sql,
            explanation=explanation if include_explanation else "",
            confidence=self._estimate_confidence(sql),
            tokens_used=response['tokens'],
            latency=latency,
            model=model,
            assumptions=assumptions
        )

    def _call_openai(self, prompt: str, model: str) -> Dict:
        """Call OpenAI API with function calling for structured output."""

        function_schema = {
            "name": "generate_sql",
            "description": "Generate SQL query from natural language question",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query (SELECT statement only)"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Natural language explanation of the query"
                    },
                    "assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of assumptions made"
                    }
                },
                "required": ["sql", "explanation"]
            }
        }

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PromptTemplates.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            functions=[function_schema],
            function_call={"name": "generate_sql"},
            temperature=0.0  # Deterministic output
        )

        # Extract function call arguments
        import json
        function_args = json.loads(
            response.choices[0].message.function_call.arguments
        )

        return {
            "content": function_args,
            "tokens": response.usage.total_tokens
        }

    def _call_anthropic(self, prompt: str, model: str) -> Dict:
        """Call Anthropic Claude API."""

        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system=PromptTemplates.SYSTEM_MESSAGE,
            temperature=0.0
        )

        # Parse structured output from text
        content = response.content[0].text

        return {
            "content": {"sql": content, "explanation": "", "assumptions": []},
            "tokens": response.usage.input_tokens + response.usage.output_tokens
        }

    def _parse_response(self, content: Dict) -> Tuple[str, str, List[str]]:
        """
        Parse LLM response to extract SQL, explanation, and assumptions.

        Args:
            content: Response content from LLM

        Returns:
            Tuple of (sql, explanation, assumptions)
        """
        sql = content.get("sql", "").strip()
        explanation = content.get("explanation", "").strip()
        assumptions = content.get("assumptions", [])

        # Clean up SQL
        sql = sql.replace("```sql", "").replace("```", "").strip()

        # Remove trailing semicolon if present
        if sql.endswith(";"):
            sql = sql[:-1]

        return sql, explanation, assumptions

    def _classify_complexity(self, question: str) -> str:
        """
        Classify query complexity: simple, medium, or complex.

        Args:
            question: Natural language question

        Returns:
            Complexity level ('simple', 'medium', 'complex')
        """
        question_lower = question.lower()
        word_count = len(question.split())

        # Simple: Short questions, basic filtering
        if word_count < 10 and 'join' not in question_lower:
            return 'simple'

        # Complex: Advanced SQL features
        complex_indicators = [
            'subquery', 'window function', 'recursive', 'pivot',
            'month over month', 'year over year', 'moving average',
            'percentile', 'rank'
        ]

        if any(indicator in question_lower for indicator in complex_indicators):
            return 'complex'

        # Medium: Everything else
        return 'medium'

    def _select_model(self, complexity: str) -> str:
        """
        Select appropriate model based on query complexity.

        Args:
            complexity: Query complexity level

        Returns:
            Model name
        """
        if self.provider == "openai":
            model_map = {
                'simple': 'gpt-3.5-turbo',  # Cheaper for simple queries
                'medium': 'gpt-4',
                'complex': 'gpt-4'
            }
        else:  # anthropic
            model_map = {
                'simple': 'claude-3-sonnet-20240229',
                'medium': 'claude-3-opus-20240229',
                'complex': 'claude-3-opus-20240229'
            }

        return model_map[complexity]

    def _estimate_confidence(self, sql: str) -> float:
        """
        Estimate confidence in generated SQL.

        This is a heuristic based on query structure.

        Args:
            sql: Generated SQL query

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 1.0

        # Lower confidence for very long queries
        if len(sql) > 500:
            confidence -= 0.1

        # Lower confidence for complex subqueries
        if sql.upper().count('SELECT') > 2:
            confidence -= 0.15

        # Lower confidence for window functions
        if 'OVER' in sql.upper():
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _get_cache_key(self, question: str) -> str:
        """
        Generate cache key for question.

        Args:
            question: Natural language question

        Returns:
            Cache key (MD5 hash)
        """
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    def execute_query(self, sql: str, limit: int = 1000) -> Dict:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query to execute
            limit: Maximum number of rows to return

        Returns:
            Dictionary with query results

        Raises:
            ValueError: If SQL is invalid
            RuntimeError: If query execution fails
        """
        # Validate first
        is_valid, error_msg = self.validator.validate(sql)
        if not is_valid:
            raise ValueError(f"Invalid SQL: {error_msg}")

        # Execute
        return self.validator.execute(sql, limit=limit)

    def clear_cache(self):
        """Clear the query cache."""
        if self.cache is not None:
            self.cache.clear()

    def get_stats(self) -> Dict:
        """
        Get statistics about generator usage.

        Returns:
            Dictionary with statistics
        """
        return {
            "cache_size": len(self.cache) if self.cache else 0,
            "provider": self.provider,
            "model": self.model,
            "tables_in_schema": len(self.schema_manager.get_all_tables())
        }
