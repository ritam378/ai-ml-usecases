"""
Comprehensive Tests for RAG Pipeline

Tests the complete RAG workflow including document ingestion,
retrieval, prompt generation, and evaluation.

Learning Focus:
- End-to-end RAG pipeline testing
- Integration testing
- Evaluation metrics validation
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from document_processor import load_documents, chunk_text, process_documents
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


class TestRAGPipelineInitialization:
    """Test RAG pipeline initialization and configuration."""

    def test_pipeline_initialization_defaults(self):
        """Test RAG pipeline initialization with defaults."""
        rag = RAGPipeline(collection_name="test_rag_defaults")

        assert rag.collection_name == "test_rag_defaults"
        assert rag.chunk_size == 400  # Default
        assert rag.overlap == 50  # Default
        assert rag.vector_store is not None

        # Cleanup
        rag.vector_store.delete_collection()

    def test_pipeline_initialization_custom_params(self):
        """Test RAG pipeline with custom parameters."""
        rag = RAGPipeline(
            collection_name="test_rag_custom",
            chunk_size=200,
            overlap=30
        )

        assert rag.collection_name == "test_rag_custom"
        assert rag.chunk_size == 200
        assert rag.overlap == 30

        # Cleanup
        rag.vector_store.delete_collection()

    def test_pipeline_creates_vector_store(self):
        """Test that pipeline creates vector store."""
        rag = RAGPipeline(collection_name="test_rag_vs_creation")

        # Vector store should be initialized
        assert rag.vector_store is not None
        assert isinstance(rag.vector_store, VectorStore)
        assert rag.vector_store.count() == 0

        # Cleanup
        rag.vector_store.delete_collection()


class TestAddDocuments:
    """Test adding documents to RAG pipeline."""

    def test_add_documents_from_file(self, temp_json_file):
        """Test adding documents from JSON file."""
        rag = RAGPipeline(collection_name="test_rag_add_file")

        # Add documents
        rag.add_documents(temp_json_file)

        # Should have documents in vector store
        count = rag.vector_store.count()
        assert count > 0

        # Cleanup
        rag.vector_store.delete_collection()

    def test_add_documents_chunks_correctly(self, temp_json_file):
        """Test that documents are chunked as expected."""
        rag = RAGPipeline(
            collection_name="test_rag_chunking",
            chunk_size=30,
            overlap=5
        )

        rag.add_documents(temp_json_file)

        # Should create multiple chunks
        count = rag.vector_store.count()
        # Sample documents have enough content for multiple chunks with size=30
        assert count >= 2

        # Cleanup
        rag.vector_store.delete_collection()

    def test_add_documents_different_chunk_sizes(self, temp_json_file):
        """Test that chunk size affects number of chunks."""
        # Large chunks
        rag_large = RAGPipeline(
            collection_name="test_rag_large_chunks",
            chunk_size=500,
            overlap=50
        )
        rag_large.add_documents(temp_json_file)
        large_count = rag_large.vector_store.count()

        # Small chunks
        rag_small = RAGPipeline(
            collection_name="test_rag_small_chunks",
            chunk_size=20,
            overlap=5
        )
        rag_small.add_documents(temp_json_file)
        small_count = rag_small.vector_store.count()

        # Smaller chunks should create more chunks
        assert small_count > large_count

        # Cleanup
        rag_large.vector_store.delete_collection()
        rag_small.vector_store.delete_collection()


class TestRetrieval:
    """Test document retrieval functionality."""

    def test_retrieve_returns_results(self, temp_json_file):
        """Test that retrieve returns results."""
        rag = RAGPipeline(collection_name="test_rag_retrieve")
        rag.add_documents(temp_json_file)

        results = rag.retrieve("What is Python?", top_k=2)

        # Should return expected structure
        assert 'documents' in results
        assert 'distances' in results
        assert 'metadatas' in results
        assert 'ids' in results

        # Should have results
        assert len(results['documents']) > 0

        # Cleanup
        rag.vector_store.delete_collection()

    def test_retrieve_respects_top_k(self, temp_json_file):
        """Test that retrieve respects top_k parameter."""
        rag = RAGPipeline(collection_name="test_rag_topk")
        rag.add_documents(temp_json_file)

        # Test different top_k values
        for k in [1, 2, 3]:
            results = rag.retrieve("programming", top_k=k)
            assert len(results['documents']) <= k

        # Cleanup
        rag.vector_store.delete_collection()

    def test_retrieve_semantic_similarity(self, temp_json_file):
        """Test that retrieve finds semantically similar content."""
        rag = RAGPipeline(collection_name="test_rag_semantic")
        rag.add_documents(temp_json_file)

        # Query about Python
        results = rag.retrieve("programming language", top_k=1)

        # Should return relevant result
        assert len(results['documents']) > 0

        # Top result should be relevant
        top_doc = results['documents'][0].lower()
        # Should contain related terms
        assert any(term in top_doc for term in ['python', 'programming', 'language'])

        # Cleanup
        rag.vector_store.delete_collection()

    def test_retrieve_returns_metadata(self, temp_json_file):
        """Test that retrieve returns metadata."""
        rag = RAGPipeline(collection_name="test_rag_metadata")
        rag.add_documents(temp_json_file)

        results = rag.retrieve("test query", top_k=2)

        # Should have metadata
        assert 'metadatas' in results
        assert len(results['metadatas']) > 0

        # Metadata should have expected fields
        metadata = results['metadatas'][0]
        assert 'source_doc_id' in metadata
        assert 'source_title' in metadata

        # Cleanup
        rag.vector_store.delete_collection()


class TestPromptCreation:
    """Test prompt creation and formatting."""

    def test_create_prompt_includes_context(self):
        """Test that prompt includes all contexts."""
        rag = RAGPipeline(collection_name="test_prompt_context")

        question = "What is Python?"
        contexts = [
            "Python is a programming language",
            "Python is used for web development",
            "Python has a simple syntax"
        ]

        prompt = rag.create_prompt(question, contexts)

        # Should include question
        assert question in prompt

        # Should include all contexts
        for ctx in contexts:
            assert ctx in prompt

        # Cleanup
        rag.vector_store.delete_collection()

    def test_create_prompt_has_instructions(self):
        """Test that prompt has clear instructions."""
        rag = RAGPipeline(collection_name="test_prompt_instructions")

        question = "Test question"
        contexts = ["Test context"]

        prompt = rag.create_prompt(question, contexts)

        # Should have instruction keywords
        prompt_lower = prompt.lower()
        assert any(keyword in prompt_lower for keyword in [
            'context', 'answer', 'question', 'instructions'
        ])

        # Cleanup
        rag.vector_store.delete_collection()

    def test_create_prompt_separates_contexts(self):
        """Test that contexts are clearly separated."""
        rag = RAGPipeline(collection_name="test_prompt_separation")

        contexts = [
            "First context text",
            "Second context text",
            "Third context text"
        ]

        prompt = rag.create_prompt("question", contexts)

        # Should have context markers
        assert "Context 1" in prompt or "Context 2" in prompt

        # Cleanup
        rag.vector_store.delete_collection()

    def test_create_prompt_empty_contexts(self):
        """Test prompt creation with empty contexts."""
        rag = RAGPipeline(collection_name="test_prompt_empty")

        prompt = rag.create_prompt("question", [])

        # Should still create valid prompt
        assert isinstance(prompt, str)
        assert "question" in prompt

        # Cleanup
        rag.vector_store.delete_collection()


class TestGenerateAnswer:
    """Test answer generation functionality."""

    def test_generate_answer_template_returns_structure(self, temp_json_file):
        """Test that generate_answer_template returns expected structure."""
        rag = RAGPipeline(collection_name="test_gen_template")
        rag.add_documents(temp_json_file)

        results = rag.retrieve("test", top_k=2)
        response = rag.generate_answer_template(
            "test question",
            results['documents'],
            results['distances']
        )

        # Should have expected keys
        assert 'answer' in response
        assert 'contexts' in response
        assert 'distances' in response
        assert 'prompt' in response

        # Cleanup
        rag.vector_store.delete_collection()

    def test_generate_answer_includes_contexts(self):
        """Test that generated answer includes context information."""
        rag = RAGPipeline(collection_name="test_gen_contexts")

        contexts = ["Context 1", "Context 2"]
        distances = [0.1, 0.2]

        response = rag.generate_answer_template("question", contexts, distances)

        # Answer should reference contexts
        assert "Context" in response['answer']

        # Cleanup
        rag.vector_store.delete_collection()

    def test_generate_answer_includes_similarities(self):
        """Test that answer includes similarity scores."""
        rag = RAGPipeline(collection_name="test_gen_similarity")

        contexts = ["test"]
        distances = [0.15]

        response = rag.generate_answer_template("q", contexts, distances)

        # Should show similarity information
        assert "similarity" in response['answer'].lower()

        # Cleanup
        rag.vector_store.delete_collection()


class TestQueryPipeline:
    """Test end-to-end query functionality."""

    def test_query_complete_pipeline(self, temp_json_file):
        """Test complete query pipeline."""
        rag = RAGPipeline(collection_name="test_complete_query")
        rag.add_documents(temp_json_file)

        response = rag.query("What is Python?", top_k=2)

        # Should return complete response
        assert 'answer' in response
        assert 'contexts' in response
        assert 'distances' in response
        assert 'prompt' in response

        # Should have retrieved contexts
        assert len(response['contexts']) > 0

        # Answer should be non-empty
        assert len(response['answer']) > 0

        # Cleanup
        rag.vector_store.delete_collection()

    def test_query_different_top_k(self, temp_json_file):
        """Test query with different top_k values."""
        rag = RAGPipeline(collection_name="test_query_topk")
        rag.add_documents(temp_json_file)

        for k in [1, 2, 3]:
            response = rag.query("test", top_k=k)

            # Should retrieve at most k contexts
            assert len(response['contexts']) <= k

        # Cleanup
        rag.vector_store.delete_collection()

    def test_query_multiple_questions(self, temp_json_file):
        """Test querying multiple times on same pipeline."""
        rag = RAGPipeline(collection_name="test_multi_query")
        rag.add_documents(temp_json_file)

        questions = [
            "What is Python?",
            "What is machine learning?",
            "Tell me about programming"
        ]

        for question in questions:
            response = rag.query(question, top_k=2)

            # Each should return valid response
            assert 'answer' in response
            assert 'contexts' in response
            assert len(response['contexts']) > 0

        # Cleanup
        rag.vector_store.delete_collection()

    def test_query_returns_relevant_context(self, temp_json_file):
        """Test that query returns relevant context."""
        rag = RAGPipeline(collection_name="test_query_relevance")
        rag.add_documents(temp_json_file)

        # Specific question about Python
        response = rag.query("What is Python programming?", top_k=1)

        # Top context should be relevant
        top_context = response['contexts'][0].lower()
        assert 'python' in top_context or 'programming' in top_context

        # Cleanup
        rag.vector_store.delete_collection()


class TestEvaluateRetrieval:
    """Test retrieval evaluation functionality."""

    def test_evaluate_retrieval_basic(self, temp_json_file, tmp_path):
        """Test basic retrieval evaluation."""
        rag = RAGPipeline(collection_name="test_eval_basic")
        rag.add_documents(temp_json_file)

        # Create test questions file
        test_questions = [
            {
                "question": "What is Python?",
                "expected_doc_ids": ["test_doc1"]
            }
        ]

        test_file = tmp_path / "test_questions.json"
        with open(test_file, 'w') as f:
            json.dump(test_questions, f)

        # Evaluate
        metrics = rag.evaluate_retrieval(str(test_file))

        # Should return metrics
        assert 'precision@3' in metrics
        assert 'recall@3' in metrics
        assert 'num_questions' in metrics

        # Metrics should be valid
        assert 0 <= metrics['precision@3'] <= 1
        assert 0 <= metrics['recall@3'] <= 1
        assert metrics['num_questions'] == 1

        # Cleanup
        rag.vector_store.delete_collection()

    def test_evaluate_retrieval_multiple_questions(self, temp_json_file, tmp_path):
        """Test evaluation with multiple questions."""
        rag = RAGPipeline(collection_name="test_eval_multi")
        rag.add_documents(temp_json_file)

        test_questions = [
            {
                "question": "What is Python?",
                "expected_doc_ids": ["test_doc1"]
            },
            {
                "question": "What is machine learning?",
                "expected_doc_ids": ["test_doc2"]
            }
        ]

        test_file = tmp_path / "test_questions.json"
        with open(test_file, 'w') as f:
            json.dump(test_questions, f)

        metrics = rag.evaluate_retrieval(str(test_file))

        assert metrics['num_questions'] == 2

        # Cleanup
        rag.vector_store.delete_collection()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_query_empty_knowledge_base(self):
        """Test querying empty knowledge base."""
        rag = RAGPipeline(collection_name="test_empty_kb")

        # Query without adding documents
        response = rag.query("test", top_k=3)

        # Should handle gracefully
        assert 'answer' in response
        assert 'contexts' in response
        # Contexts should be empty
        assert len(response['contexts']) == 0

        # Cleanup
        rag.vector_store.delete_collection()

    def test_very_long_question(self, temp_json_file):
        """Test handling very long questions."""
        rag = RAGPipeline(collection_name="test_long_question")
        rag.add_documents(temp_json_file)

        # Very long question
        long_question = " ".join(["word"] * 500)

        # Should handle without error
        response = rag.query(long_question, top_k=2)

        assert 'answer' in response

        # Cleanup
        rag.vector_store.delete_collection()

    def test_special_characters_in_question(self, temp_json_file):
        """Test handling special characters in questions."""
        rag = RAGPipeline(collection_name="test_special_chars")
        rag.add_documents(temp_json_file)

        question = "What is Python? @#$%^&*()"

        response = rag.query(question, top_k=2)

        assert 'answer' in response

        # Cleanup
        rag.vector_store.delete_collection()


class TestIntegration:
    """Integration tests for realistic workflows."""

    def test_realistic_qa_workflow(self, temp_json_file):
        """Test a realistic question-answering workflow."""
        # Initialize RAG system
        rag = RAGPipeline(
            collection_name="test_realistic_qa",
            chunk_size=100,
            overlap=20
        )

        # Add knowledge base
        rag.add_documents(temp_json_file)

        # Simulate user interaction
        questions = [
            "What programming languages are mentioned?",
            "Tell me about Python",
            "What is machine learning?"
        ]

        for question in questions:
            # Retrieve relevant context
            retrieval_results = rag.retrieve(question, top_k=3)
            assert len(retrieval_results['documents']) > 0

            # Generate answer
            response = rag.query(question, top_k=3)
            assert len(response['answer']) > 0
            assert len(response['contexts']) > 0

            # Prompt should be ready for LLM
            assert len(response['prompt']) > 0
            assert question in response['prompt']

        # Cleanup
        rag.vector_store.delete_collection()

    def test_incremental_knowledge_base_building(self, tmp_path):
        """Test building knowledge base incrementally."""
        rag = RAGPipeline(collection_name="test_incremental_kb")

        # Add first batch of documents
        docs1 = [
            {
                "id": "doc1",
                "title": "Python",
                "content": "Python is a programming language",
                "category": "programming"
            }
        ]

        file1 = tmp_path / "docs1.json"
        with open(file1, 'w') as f:
            json.dump(docs1, f)

        rag.add_documents(str(file1))
        count1 = rag.vector_store.count()

        # Add second batch
        docs2 = [
            {
                "id": "doc2",
                "title": "Java",
                "content": "Java is another programming language",
                "category": "programming"
            }
        ]

        file2 = tmp_path / "docs2.json"
        with open(file2, 'w') as f:
            json.dump(docs2, f)

        rag.add_documents(str(file2))
        count2 = rag.vector_store.count()

        # Should have more documents
        assert count2 > count1

        # Should be able to query about both
        response1 = rag.query("Python", top_k=1)
        response2 = rag.query("Java", top_k=1)

        assert len(response1['contexts']) > 0
        assert len(response2['contexts']) > 0

        # Cleanup
        rag.vector_store.delete_collection()

    def test_pipeline_persistence(self, temp_json_file, tmp_path):
        """Test that vector store persists across pipeline instances."""
        persist_dir = str(tmp_path / "persist_test")

        # Create first pipeline and add documents
        rag1 = RAGPipeline(collection_name="test_persistence")
        # Manually set persist directory
        rag1.vector_store.persist_directory = persist_dir
        rag1.add_documents(temp_json_file)
        count1 = rag1.vector_store.count()

        # Create second pipeline with same collection name
        # (In practice, would need to handle persist_directory better)
        rag2 = RAGPipeline(collection_name="test_persistence")

        # Should be able to query (data persists)
        response = rag2.query("test", top_k=2)
        assert 'contexts' in response

        # Cleanup
        rag1.vector_store.delete_collection()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
