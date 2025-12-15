"""
Pytest fixtures for RAG system tests.
"""

import pytest
import json
import tempfile
import os


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "test_doc1",
            "title": "Python Basics",
            "content": "Python is a programming language. It is easy to learn and powerful. Python supports multiple programming paradigms including object-oriented and functional programming.",
            "category": "python"
        },
        {
            "id": "test_doc2",
            "title": "Machine Learning Intro",
            "content": "Machine learning is a type of artificial intelligence. It allows computers to learn from data without being explicitly programmed. Supervised learning uses labeled data.",
            "category": "ml"
        }
    ]


@pytest.fixture
def sample_chunks():
    """Sample processed chunks for testing."""
    return [
        {
            'id': 'doc1_chunk0',
            'text': 'Python is a programming language used for many applications.',
            'source_doc_id': 'doc1',
            'source_title': 'Python Guide',
            'category': 'python',
            'chunk_index': 0,
            'total_chunks': 1
        },
        {
            'id': 'doc2_chunk0',
            'text': 'Machine learning enables computers to learn patterns from data.',
            'source_doc_id': 'doc2',
            'source_title': 'ML Basics',
            'category': 'ml',
            'chunk_index': 0,
            'total_chunks': 1
        }
    ]


@pytest.fixture
def temp_json_file(sample_documents, tmp_path):
    """Create temporary JSON file with sample documents."""
    filepath = tmp_path / "test_docs.json"
    with open(filepath, 'w') as f:
        json.dump(sample_documents, f)
    return str(filepath)


@pytest.fixture
def temp_vector_store_dir(tmp_path):
    """Temporary directory for vector store."""
    return str(tmp_path / "test_chroma_db")
