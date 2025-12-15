"""
Comprehensive Tests for Vector Store

Tests vector database operations including adding documents,
semantic search, and similarity metrics.

Learning Focus:
- Vector database operations
- Semantic search validation
- Embedding quality testing
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vector_store import VectorStore


class TestVectorStoreInitialization:
    """Test vector store initialization and setup."""

    def test_create_new_collection(self, temp_vector_store_dir):
        """Test creating a new collection."""
        store = VectorStore(
            collection_name="test_new_collection",
            persist_directory=temp_vector_store_dir
        )

        assert store.collection_name == "test_new_collection"
        assert store.persist_directory == temp_vector_store_dir
        assert store.collection is not None
        assert store.count() == 0

        # Cleanup
        store.delete_collection()

    def test_load_existing_collection(self, temp_vector_store_dir, sample_chunks):
        """Test loading an existing collection."""
        # Create and populate collection
        store1 = VectorStore(
            collection_name="test_existing",
            persist_directory=temp_vector_store_dir
        )
        store1.add_documents(sample_chunks)
        initial_count = store1.count()

        # Create new instance with same collection name
        store2 = VectorStore(
            collection_name="test_existing",
            persist_directory=temp_vector_store_dir
        )

        # Should load existing data
        assert store2.count() == initial_count

        # Cleanup
        store2.delete_collection()

    def test_different_collections_isolated(self, temp_vector_store_dir, sample_chunks):
        """Test that different collections are isolated."""
        # Create two different collections
        store1 = VectorStore(
            collection_name="collection1",
            persist_directory=temp_vector_store_dir
        )
        store2 = VectorStore(
            collection_name="collection2",
            persist_directory=temp_vector_store_dir
        )

        # Add documents to only first collection
        store1.add_documents(sample_chunks)

        # Collections should have different counts
        assert store1.count() > 0
        assert store2.count() == 0

        # Cleanup
        store1.delete_collection()
        store2.delete_collection()


class TestAddDocuments:
    """Test adding documents to vector store."""

    def test_add_single_chunk(self, temp_vector_store_dir):
        """Test adding a single chunk."""
        store = VectorStore(
            collection_name="test_single",
            persist_directory=temp_vector_store_dir
        )

        chunk = {
            'id': 'test_chunk_1',
            'text': 'This is a test document about Python programming.',
            'source_doc_id': 'doc1',
            'source_title': 'Test Doc',
            'category': 'test'
        }

        store.add_documents([chunk])

        assert store.count() == 1

        # Cleanup
        store.delete_collection()

    def test_add_multiple_chunks(self, temp_vector_store_dir, sample_chunks):
        """Test adding multiple chunks."""
        store = VectorStore(
            collection_name="test_multiple",
            persist_directory=temp_vector_store_dir
        )

        store.add_documents(sample_chunks)

        assert store.count() == len(sample_chunks)

        # Cleanup
        store.delete_collection()

    def test_add_empty_list(self, temp_vector_store_dir):
        """Test adding empty list of documents."""
        store = VectorStore(
            collection_name="test_empty",
            persist_directory=temp_vector_store_dir
        )

        # Should handle gracefully
        store.add_documents([])

        assert store.count() == 0

        # Cleanup
        store.delete_collection()

    def test_add_incremental(self, temp_vector_store_dir, sample_chunks):
        """Test adding documents incrementally."""
        store = VectorStore(
            collection_name="test_incremental",
            persist_directory=temp_vector_store_dir
        )

        # Add first chunk
        store.add_documents([sample_chunks[0]])
        assert store.count() == 1

        # Add second chunk
        store.add_documents([sample_chunks[1]])
        assert store.count() == 2

        # Cleanup
        store.delete_collection()

    def test_metadata_preserved(self, temp_vector_store_dir, sample_chunks):
        """Test that metadata is preserved when adding documents."""
        store = VectorStore(
            collection_name="test_metadata",
            persist_directory=temp_vector_store_dir
        )

        store.add_documents(sample_chunks)

        # Peek at stored documents
        data = store.peek(limit=1)

        # Check metadata exists
        assert 'metadatas' in data
        assert len(data['metadatas']) > 0

        metadata = data['metadatas'][0]
        assert 'source_doc_id' in metadata
        assert 'source_title' in metadata
        assert 'category' in metadata

        # Cleanup
        store.delete_collection()


class TestQuery:
    """Test semantic search functionality."""

    def test_query_returns_results(self, temp_vector_store_dir, sample_chunks):
        """Test that query returns results."""
        store = VectorStore(
            collection_name="test_query",
            persist_directory=temp_vector_store_dir
        )
        store.add_documents(sample_chunks)

        results = store.query("programming", top_k=2)

        # Should return results
        assert 'documents' in results
        assert 'distances' in results
        assert 'metadatas' in results
        assert 'ids' in results

        assert len(results['documents']) > 0
        assert len(results['documents']) <= 2

        # Cleanup
        store.delete_collection()

    def test_query_semantic_similarity(self, temp_vector_store_dir):
        """Test that semantically similar queries return relevant results."""
        store = VectorStore(
            collection_name="test_semantic",
            persist_directory=temp_vector_store_dir
        )

        chunks = [
            {
                'id': 'python_chunk',
                'text': 'Python is a high-level programming language known for its simplicity and readability.',
                'source_doc_id': 'doc1',
                'source_title': 'Python Guide',
                'category': 'python'
            },
            {
                'id': 'cooking_chunk',
                'text': 'To bake a cake, you need flour, sugar, eggs, and butter. Mix them together and bake at 350F.',
                'source_doc_id': 'doc2',
                'source_title': 'Cooking Guide',
                'category': 'cooking'
            }
        ]

        store.add_documents(chunks)

        # Query about programming should return Python chunk
        results = store.query("What programming language is easy to learn?", top_k=1)

        # Top result should be about Python
        top_doc = results['documents'][0]
        assert 'python' in top_doc.lower() or 'programming' in top_doc.lower()

        # Cleanup
        store.delete_collection()

    def test_query_top_k(self, temp_vector_store_dir):
        """Test that query respects top_k parameter."""
        store = VectorStore(
            collection_name="test_topk",
            persist_directory=temp_vector_store_dir
        )

        # Add 5 documents
        chunks = [
            {
                'id': f'chunk_{i}',
                'text': f'This is document number {i} about various topics.',
                'source_doc_id': f'doc{i}',
                'source_title': f'Doc {i}',
                'category': 'test'
            }
            for i in range(5)
        ]

        store.add_documents(chunks)

        # Test different top_k values
        for k in [1, 2, 3]:
            results = store.query("document", top_k=k)
            assert len(results['documents']) == k

        # Cleanup
        store.delete_collection()

    def test_query_distances_ordered(self, temp_vector_store_dir, sample_chunks):
        """Test that results are ordered by distance (most similar first)."""
        store = VectorStore(
            collection_name="test_ordering",
            persist_directory=temp_vector_store_dir
        )
        store.add_documents(sample_chunks)

        results = store.query("programming language", top_k=2)

        distances = results['distances']

        # Distances should be in ascending order (smaller = more similar)
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1]

        # Cleanup
        store.delete_collection()

    def test_query_empty_collection(self, temp_vector_store_dir):
        """Test querying an empty collection."""
        store = VectorStore(
            collection_name="test_empty_query",
            persist_directory=temp_vector_store_dir
        )

        results = store.query("test query", top_k=3)

        # Should return empty results
        assert len(results['documents']) == 0
        assert len(results['distances']) == 0

        # Cleanup
        store.delete_collection()

    def test_query_more_than_available(self, temp_vector_store_dir):
        """Test requesting more results than available documents."""
        store = VectorStore(
            collection_name="test_over_request",
            persist_directory=temp_vector_store_dir
        )

        # Add only 2 documents
        chunks = [
            {
                'id': 'chunk1',
                'text': 'First document',
                'source_doc_id': 'doc1',
                'source_title': 'Doc 1',
                'category': 'test'
            },
            {
                'id': 'chunk2',
                'text': 'Second document',
                'source_doc_id': 'doc2',
                'source_title': 'Doc 2',
                'category': 'test'
            }
        ]

        store.add_documents(chunks)

        # Request 10 but should only get 2
        results = store.query("document", top_k=10)

        assert len(results['documents']) == 2

        # Cleanup
        store.delete_collection()

    def test_query_result_structure(self, temp_vector_store_dir, sample_chunks):
        """Test the structure of query results."""
        store = VectorStore(
            collection_name="test_structure",
            persist_directory=temp_vector_store_dir
        )
        store.add_documents(sample_chunks)

        results = store.query("test", top_k=1)

        # Check all expected keys exist
        assert 'documents' in results
        assert 'distances' in results
        assert 'metadatas' in results
        assert 'ids' in results

        # Check all lists have same length
        length = len(results['documents'])
        assert len(results['distances']) == length
        assert len(results['metadatas']) == length
        assert len(results['ids']) == length

        # Check types
        assert isinstance(results['documents'], list)
        assert isinstance(results['distances'], list)
        assert isinstance(results['metadatas'], list)
        assert isinstance(results['ids'], list)

        # Cleanup
        store.delete_collection()


class TestUtilityMethods:
    """Test utility methods like count, peek, delete."""

    def test_count_empty(self, temp_vector_store_dir):
        """Test count on empty collection."""
        store = VectorStore(
            collection_name="test_count_empty",
            persist_directory=temp_vector_store_dir
        )

        assert store.count() == 0

        # Cleanup
        store.delete_collection()

    def test_count_after_adding(self, temp_vector_store_dir, sample_chunks):
        """Test count after adding documents."""
        store = VectorStore(
            collection_name="test_count_after",
            persist_directory=temp_vector_store_dir
        )

        # Initially empty
        assert store.count() == 0

        # Add documents
        store.add_documents(sample_chunks)

        # Count should match
        assert store.count() == len(sample_chunks)

        # Cleanup
        store.delete_collection()

    def test_peek_returns_data(self, temp_vector_store_dir, sample_chunks):
        """Test peek returns sample data."""
        store = VectorStore(
            collection_name="test_peek",
            persist_directory=temp_vector_store_dir
        )
        store.add_documents(sample_chunks)

        data = store.peek(limit=2)

        # Should have data
        assert 'documents' in data or 'ids' in data

        # Cleanup
        store.delete_collection()

    def test_peek_respects_limit(self, temp_vector_store_dir):
        """Test that peek respects limit parameter."""
        store = VectorStore(
            collection_name="test_peek_limit",
            persist_directory=temp_vector_store_dir
        )

        # Add 5 documents
        chunks = [
            {
                'id': f'chunk_{i}',
                'text': f'Document {i}',
                'source_doc_id': f'doc{i}',
                'source_title': f'Doc {i}',
                'category': 'test'
            }
            for i in range(5)
        ]

        store.add_documents(chunks)

        # Peek with limit 2
        data = store.peek(limit=2)

        # Should return at most 2
        if 'ids' in data:
            assert len(data['ids']) <= 2

        # Cleanup
        store.delete_collection()

    def test_delete_collection(self, temp_vector_store_dir, sample_chunks):
        """Test deleting a collection."""
        store = VectorStore(
            collection_name="test_delete",
            persist_directory=temp_vector_store_dir
        )

        store.add_documents(sample_chunks)
        assert store.count() > 0

        # Delete collection
        store.delete_collection()

        # Create new store with same name - should be empty
        new_store = VectorStore(
            collection_name="test_delete",
            persist_directory=temp_vector_store_dir
        )

        assert new_store.count() == 0

        # Cleanup
        new_store.delete_collection()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_text(self, temp_vector_store_dir):
        """Test handling very long text."""
        store = VectorStore(
            collection_name="test_long_text",
            persist_directory=temp_vector_store_dir
        )

        # Create very long text (5000 words)
        long_text = ' '.join(['word'] * 5000)

        chunk = {
            'id': 'long_chunk',
            'text': long_text,
            'source_doc_id': 'doc1',
            'source_title': 'Long Doc',
            'category': 'test'
        }

        # Should handle without error
        store.add_documents([chunk])
        assert store.count() == 1

        # Cleanup
        store.delete_collection()

    def test_special_characters_in_text(self, temp_vector_store_dir):
        """Test handling special characters."""
        store = VectorStore(
            collection_name="test_special_chars",
            persist_directory=temp_vector_store_dir
        )

        chunk = {
            'id': 'special_chunk',
            'text': 'Text with special chars: @#$%^&*(){}[]|\\:";\'<>?,./~`',
            'source_doc_id': 'doc1',
            'source_title': 'Special Doc',
            'category': 'test'
        }

        store.add_documents([chunk])
        assert store.count() == 1

        # Should be queryable
        results = store.query("special", top_k=1)
        assert len(results['documents']) > 0

        # Cleanup
        store.delete_collection()

    def test_unicode_text(self, temp_vector_store_dir):
        """Test handling Unicode characters."""
        store = VectorStore(
            collection_name="test_unicode",
            persist_directory=temp_vector_store_dir
        )

        chunk = {
            'id': 'unicode_chunk',
            'text': 'Text with Unicode: 你好 مرحبا привет 🚀 café résumé',
            'source_doc_id': 'doc1',
            'source_title': 'Unicode Doc',
            'category': 'test'
        }

        store.add_documents([chunk])
        assert store.count() == 1

        # Cleanup
        store.delete_collection()

    def test_empty_text(self, temp_vector_store_dir):
        """Test handling empty text."""
        store = VectorStore(
            collection_name="test_empty_text",
            persist_directory=temp_vector_store_dir
        )

        chunk = {
            'id': 'empty_chunk',
            'text': '',
            'source_doc_id': 'doc1',
            'source_title': 'Empty Doc',
            'category': 'test'
        }

        # Should handle empty text
        store.add_documents([chunk])

        # Cleanup
        store.delete_collection()


class TestIntegration:
    """Integration tests for realistic workflows."""

    def test_realistic_rag_workflow(self, temp_vector_store_dir):
        """Test a realistic RAG workflow."""
        store = VectorStore(
            collection_name="test_realistic",
            persist_directory=temp_vector_store_dir
        )

        # Simulate adding knowledge base articles
        articles = [
            {
                'id': 'python_intro',
                'text': 'Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple programming paradigms.',
                'source_doc_id': 'doc1',
                'source_title': 'Python Introduction',
                'category': 'programming'
            },
            {
                'id': 'ml_basics',
                'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.',
                'source_doc_id': 'doc2',
                'source_title': 'ML Basics',
                'category': 'ai'
            },
            {
                'id': 'web_dev',
                'text': 'Web development involves creating websites and web applications. Common technologies include HTML, CSS, and JavaScript.',
                'source_doc_id': 'doc3',
                'source_title': 'Web Development',
                'category': 'web'
            }
        ]

        # Add all articles
        store.add_documents(articles)

        # Simulate user questions
        questions = [
            "What is Python?",
            "Tell me about machine learning",
            "How do I build websites?"
        ]

        for question in questions:
            results = store.query(question, top_k=2)

            # Should return relevant results
            assert len(results['documents']) > 0
            assert all(isinstance(doc, str) for doc in results['documents'])
            assert all(isinstance(dist, float) for dist in results['distances'])

        # Cleanup
        store.delete_collection()

    def test_multi_document_retrieval(self, temp_vector_store_dir):
        """Test retrieving from multiple related documents."""
        store = VectorStore(
            collection_name="test_multi_doc",
            persist_directory=temp_vector_store_dir
        )

        # Add chunks from same topic but different documents
        chunks = [
            {
                'id': 'py_chunk1',
                'text': 'Python was created by Guido van Rossum in 1991.',
                'source_doc_id': 'python_history',
                'source_title': 'Python History',
                'category': 'python'
            },
            {
                'id': 'py_chunk2',
                'text': 'Python is widely used in data science and machine learning.',
                'source_doc_id': 'python_applications',
                'source_title': 'Python Applications',
                'category': 'python'
            },
            {
                'id': 'py_chunk3',
                'text': 'Python has a simple, easy-to-learn syntax that emphasizes readability.',
                'source_doc_id': 'python_features',
                'source_title': 'Python Features',
                'category': 'python'
            }
        ]

        store.add_documents(chunks)

        # Query should potentially retrieve from multiple documents
        results = store.query("Tell me about Python", top_k=3)

        assert len(results['documents']) == 3

        # Should have results from multiple source documents
        source_docs = [meta['source_doc_id'] for meta in results['metadatas']]
        # At least 2 different sources (likely all 3)
        assert len(set(source_docs)) >= 2

        # Cleanup
        store.delete_collection()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
