"""
Prompt Templates for Text-to-SQL System

This module contains prompt templates for LLM-based SQL generation.
"""

from typing import List, Dict


class PromptTemplates:
    """Collection of prompt templates for Text-to-SQL generation."""

    # System message defining the LLM's role
    SYSTEM_MESSAGE = """You are an expert SQL developer. Your task is to generate valid SQL queries based on natural language questions and database schemas.

Guidelines:
1. Generate ONLY valid, executable SQL queries
2. Use proper JOIN syntax when combining tables
3. Always specify column names explicitly (never use SELECT *)
4. Include appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses
5. Handle NULLs appropriately
6. Use table aliases for clarity when joining multiple tables
7. Optimize for readability and correctness
8. For aggregations, include all non-aggregated columns in GROUP BY

Database Dialect: SQLite

Output Requirements:
- Return syntactically correct SQL
- Use standard SQL syntax
- Avoid database-specific extensions unless necessary
- Add LIMIT clause to prevent large result sets (default LIMIT 100)"""

    # Few-shot examples for common query patterns
    FEW_SHOT_EXAMPLES = [
        {
            "question": "Show all customers from New York",
            "sql": """SELECT id, name, email, city
FROM customers
WHERE city = 'New York'
LIMIT 100""",
            "explanation": "Simple filter query on the customers table"
        },
        {
            "question": "What are the top 5 customers by total order value?",
            "sql": """SELECT c.id, c.name, SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY total_spent DESC
LIMIT 5""",
            "explanation": "Join customers and orders, aggregate by customer, sort descending"
        },
        {
            "question": "Find products that have never been ordered",
            "sql": """SELECT p.id, p.name, p.category, p.price
FROM products p
LEFT JOIN order_items oi ON p.id = oi.product_id
WHERE oi.product_id IS NULL
LIMIT 100""",
            "explanation": "Left join to find products with no matching order items"
        },
        {
            "question": "Show average order value by month for 2024",
            "sql": """SELECT
    strftime('%Y-%m', order_date) as month,
    COUNT(*) as order_count,
    AVG(total_amount) as avg_order_value,
    SUM(total_amount) as total_revenue
FROM orders
WHERE strftime('%Y', order_date) = '2024'
GROUP BY strftime('%Y-%m', order_date)
ORDER BY month""",
            "explanation": "Date functions for monthly aggregation with multiple metrics"
        },
        {
            "question": "Which products are most frequently purchased together?",
            "sql": """SELECT
    oi1.product_id as product_1,
    oi2.product_id as product_2,
    COUNT(*) as times_bought_together
FROM order_items oi1
JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
GROUP BY oi1.product_id, oi2.product_id
ORDER BY times_bought_together DESC
LIMIT 20""",
            "explanation": "Self-join to find co-purchased products (market basket analysis)"
        }
    ]

    @staticmethod
    def build_prompt(
        schema: str,
        question: str,
        examples: List[Dict] = None,
        include_explanation: bool = True
    ) -> str:
        """
        Build complete prompt for SQL generation.

        Args:
            schema: Formatted database schema
            question: Natural language question
            examples: List of few-shot examples (use defaults if None)
            include_explanation: Whether to ask for explanation

        Returns:
            Complete prompt string
        """
        if examples is None:
            examples = PromptTemplates.FEW_SHOT_EXAMPLES[:3]  # Use first 3 examples

        prompt_parts = []

        # Schema section
        prompt_parts.append("# Database Schema\n")
        prompt_parts.append(schema)
        prompt_parts.append("\n")

        # Examples section
        if examples:
            prompt_parts.append("# Example Questions and SQL Queries\n")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\n## Example {i}")
                prompt_parts.append(f"Question: {example['question']}")
                prompt_parts.append(f"SQL:\n{example['sql']}")
                if 'explanation' in example:
                    prompt_parts.append(f"Explanation: {example['explanation']}")
                prompt_parts.append("")

        # Question section
        prompt_parts.append("# Your Task\n")
        prompt_parts.append(f"Question: {question}\n")

        # Instructions
        prompt_parts.append("Generate a SQL query to answer this question.")

        if include_explanation:
            prompt_parts.append("\nProvide:")
            prompt_parts.append("1. SQL query (syntactically correct and executable)")
            prompt_parts.append("2. Brief explanation of the query logic")
            prompt_parts.append("3. Any assumptions you made")
        else:
            prompt_parts.append("\nProvide only the SQL query (syntactically correct and executable).")

        return "\n".join(prompt_parts)

    @staticmethod
    def build_retry_prompt(
        original_question: str,
        failed_sql: str,
        error_message: str
    ) -> str:
        """
        Build prompt for retry after error.

        Args:
            original_question: Original natural language question
            failed_sql: SQL that failed
            error_message: Error message from failed execution

        Returns:
            Retry prompt with error context
        """
        return f"""The previous SQL query had an error. Please fix it.

Original Question: {original_question}

Failed SQL Query:
{failed_sql}

Error Message:
{error_message}

Please generate a corrected SQL query that fixes this error.
Provide only the corrected SQL query."""

    @staticmethod
    def build_clarification_prompt(
        question: str,
        ambiguities: List[str]
    ) -> str:
        """
        Build prompt asking for clarification.

        Args:
            question: Original question
            ambiguities: List of ambiguous aspects

        Returns:
            Clarification prompt
        """
        ambiguity_text = "\n".join(f"- {amb}" for amb in ambiguities)

        return f"""The following question has some ambiguities:

Question: {question}

Ambiguities:
{ambiguity_text}

Please clarify:
1. Which interpretation should I use?
2. Or, provide more specific details in your question.

I'll wait for your clarification before generating the SQL query."""

    @staticmethod
    def format_few_shot_examples(examples: List[Dict]) -> str:
        """
        Format few-shot examples for display.

        Args:
            examples: List of example dictionaries

        Returns:
            Formatted examples string
        """
        formatted = []

        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Question: {example['question']}")
            formatted.append(f"SQL: {example['sql']}")
            if 'explanation' in example:
                formatted.append(f"Explanation: {example['explanation']}")
            formatted.append("")

        return "\n".join(formatted)

    @staticmethod
    def get_examples_for_pattern(pattern: str) -> List[Dict]:
        """
        Get examples matching a specific pattern.

        Args:
            pattern: Query pattern (e.g., 'join', 'aggregation', 'subquery')

        Returns:
            List of relevant examples
        """
        pattern_keywords = {
            'join': ['join', 'combine', 'relate', 'together'],
            'aggregation': ['total', 'sum', 'average', 'count', 'max', 'min', 'group'],
            'filter': ['where', 'filter', 'condition', 'specific'],
            'subquery': ['subquery', 'nested', 'in', 'exists'],
            'date': ['date', 'month', 'year', 'day', 'time'],
            'null': ['null', 'missing', 'empty', 'no', 'never']
        }

        keywords = pattern_keywords.get(pattern.lower(), [])
        if not keywords:
            return PromptTemplates.FEW_SHOT_EXAMPLES[:3]

        # Find examples matching keywords
        matching_examples = []

        for example in PromptTemplates.FEW_SHOT_EXAMPLES:
            question_lower = example['question'].lower()
            sql_lower = example['sql'].lower()

            if any(kw in question_lower or kw in sql_lower for kw in keywords):
                matching_examples.append(example)

        # Return matching examples or default to first 3
        return matching_examples[:3] if matching_examples else PromptTemplates.FEW_SHOT_EXAMPLES[:3]


# Additional template variations
SIMPLE_PROMPT_TEMPLATE = """Given this database schema:

{schema}

Generate a SQL query to answer this question: {question}

Return only the SQL query, nothing else."""


DETAILED_PROMPT_TEMPLATE = """You are a SQL expert. Given the following database schema and question, generate a SQL query.

Database Schema:
{schema}

Question: {question}

Requirements:
1. Write syntactically correct SQLite SQL
2. Use proper JOIN syntax
3. Include appropriate filters, grouping, and ordering
4. Add LIMIT clause to prevent large results
5. Handle NULL values appropriately

Provide your response in this format:
SQL: <your sql query>
Explanation: <brief explanation>
Assumptions: <any assumptions made>"""


COT_PROMPT_TEMPLATE = """Given this database schema:

{schema}

Question: {question}

Let's solve this step by step:
1. First, identify which tables we need
2. Determine what columns to select
3. Figure out any JOIN conditions
4. Add WHERE filters if needed
5. Determine GROUP BY and ORDER BY clauses
6. Add LIMIT for safety

Now, generate the SQL query following these steps."""
