"""
Comprehensive Tests for Document Processor

Tests all document processing functionality including loading,
cleaning, chunking, and metadata preservation.

Learning Focus:
- Test-driven development
- Edge case handling
- Chunking strategy validation
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from document_processor import (
    load_documents,
    clean_text,
    chunk_text,
    process_documents
)


class TestLoadDocuments:
    """Test document loading functionality."""

    def test_load_valid_json(self, temp_json_file):
        """Test loading valid JSON file."""
        docs = load_documents(temp_json_file)

        assert isinstance(docs, list)
        assert len(docs) == 2
        assert all('content' in doc for doc in docs)
        assert all('id' in doc for doc in docs)

    def test_load_documents_preserves_metadata(self, temp_json_file):
        """Test that all document metadata is preserved."""
        docs = load_documents(temp_json_file)

        # Check first document has all expected fields
        doc = docs[0]
        assert 'id' in doc
        assert 'title' in doc
        assert 'content' in doc
        assert 'category' in doc

    def test_load_empty_file(self, tmp_path):
        """Test loading empty JSON array."""
        filepath = tmp_path / "empty.json"
        with open(filepath, 'w') as f:
            json.dump([], f)

        docs = load_documents(str(filepath))
        assert docs == []

    def test_load_nonexistent_file(self):
        """Test loading file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_documents("nonexistent_file.json")


class TestCleanText:
    """Test text cleaning functionality."""

    def test_clean_extra_whitespace(self):
        """Test removing extra whitespace."""
        text = "This  has   extra    spaces"
        cleaned = clean_text(text)
        assert cleaned == "This has extra spaces"

    def test_clean_leading_trailing_whitespace(self):
        """Test removing leading/trailing whitespace."""
        text = "  \n  Text with padding  \n  "
        cleaned = clean_text(text)
        assert cleaned == "Text with padding"

    def test_clean_newlines(self):
        """Test normalizing newlines."""
        text = "Line 1\n\nLine 2\n\n\nLine 3"
        cleaned = clean_text(text)
        # Multiple newlines become single spaces
        assert "\n\n" not in cleaned

    def test_clean_empty_string(self):
        """Test cleaning empty string."""
        cleaned = clean_text("")
        assert cleaned == ""

    def test_clean_tabs(self):
        """Test handling tabs."""
        text = "Text\twith\ttabs"
        cleaned = clean_text(text)
        assert "\t" not in cleaned

    def test_clean_preserves_content(self):
        """Test that cleaning preserves actual content."""
        text = "  Python is a programming language.  "
        cleaned = clean_text(text)
        assert "Python" in cleaned
        assert "programming language" in cleaned


class TestChunkText:
    """Test text chunking functionality."""

    def test_chunk_small_text(self):
        """Test chunking text smaller than chunk size."""
        text = "Short text here"
        chunks = chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_exact_size(self):
        """Test text exactly chunk_size words."""
        words = ["word"] * 50
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=50, overlap=0)

        assert len(chunks) == 1

    def test_chunk_creates_overlap(self):
        """Test that chunks have proper overlap."""
        # Create text with 100 words
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=40, overlap=10)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Check overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_words = chunks[i].split()
            chunk2_words = chunks[i + 1].split()

            # Last words of chunk1 should appear in chunk2
            overlap_words = chunk1_words[-10:]  # Last 10 words
            # At least some overlap should exist
            assert any(word in chunk2_words for word in overlap_words)

    def test_chunk_size_respected(self):
        """Test that chunks don't exceed chunk_size (except last)."""
        words = ["word"] * 200
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=50, overlap=10)

        # All chunks except possibly the last should be exactly chunk_size
        for chunk in chunks[:-1]:
            chunk_words = chunk.split()
            assert len(chunk_words) == 50

    def test_chunk_no_overlap(self):
        """Test chunking with no overlap."""
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)

        chunks = chunk_text(text, chunk_size=25, overlap=0)

        # Should have exactly 4 chunks
        assert len(chunks) == 4

        # Each should be exactly 25 words
        for chunk in chunks:
            assert len(chunk.split()) == 25

    def test_chunk_large_overlap(self):
        """Test chunking with large overlap."""
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)

        # Overlap almost as large as chunk_size
        chunks = chunk_text(text, chunk_size=40, overlap=35)

        # Should create many chunks due to small step size
        assert len(chunks) > 10

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100, overlap=10)

        # Empty text returns single empty chunk
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_chunk_single_word(self):
        """Test chunking single word."""
        chunks = chunk_text("word", chunk_size=10, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == "word"


class TestProcessDocuments:
    """Test complete document processing pipeline."""

    def test_process_single_document(self, sample_documents):
        """Test processing a single document."""
        # Use only first document
        docs = [sample_documents[0]]

        chunks = process_documents(docs, chunk_size=50, overlap=10)

        # Should create at least one chunk
        assert len(chunks) >= 1

        # Each chunk should have required fields
        for chunk in chunks:
            assert 'id' in chunk
            assert 'text' in chunk
            assert 'source_doc_id' in chunk
            assert 'source_title' in chunk
            assert 'category' in chunk
            assert 'chunk_index' in chunk
            assert 'total_chunks' in chunk

    def test_process_multiple_documents(self, sample_documents):
        """Test processing multiple documents."""
        chunks = process_documents(sample_documents, chunk_size=50, overlap=10)

        # Should create at least as many chunks as documents
        assert len(chunks) >= len(sample_documents)

        # Should have chunks from both documents
        source_ids = set(chunk['source_doc_id'] for chunk in chunks)
        assert len(source_ids) == len(sample_documents)

    def test_process_preserves_metadata(self, sample_documents):
        """Test that processing preserves document metadata."""
        chunks = process_documents(sample_documents, chunk_size=50, overlap=10)

        # Check metadata from first document is preserved
        doc = sample_documents[0]
        doc_chunks = [c for c in chunks if c['source_doc_id'] == doc['id']]

        assert len(doc_chunks) > 0
        for chunk in doc_chunks:
            assert chunk['source_doc_id'] == doc['id']
            assert chunk['source_title'] == doc['title']
            assert chunk['category'] == doc['category']

    def test_process_chunk_ids_unique(self, sample_documents):
        """Test that all chunk IDs are unique."""
        chunks = process_documents(sample_documents, chunk_size=50, overlap=10)

        chunk_ids = [chunk['id'] for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_process_chunk_indices_sequential(self, sample_documents):
        """Test that chunk indices are sequential per document."""
        chunks = process_documents(sample_documents, chunk_size=30, overlap=5)

        # Group chunks by source document
        for doc in sample_documents:
            doc_chunks = [c for c in chunks if c['source_doc_id'] == doc['id']]
            doc_chunks.sort(key=lambda x: x['chunk_index'])

            # Indices should be 0, 1, 2, ...
            expected_indices = list(range(len(doc_chunks)))
            actual_indices = [c['chunk_index'] for c in doc_chunks]
            assert actual_indices == expected_indices

    def test_process_total_chunks_correct(self, sample_documents):
        """Test that total_chunks field is correct."""
        chunks = process_documents(sample_documents, chunk_size=30, overlap=5)

        for doc in sample_documents:
            doc_chunks = [c for c in chunks if c['source_doc_id'] == doc['id']]

            # All chunks from same document should have same total_chunks
            total_chunks_values = [c['total_chunks'] for c in doc_chunks]
            assert len(set(total_chunks_values)) == 1

            # total_chunks should match actual number of chunks
            assert doc_chunks[0]['total_chunks'] == len(doc_chunks)

    def test_process_empty_documents(self):
        """Test processing empty document list."""
        chunks = process_documents([], chunk_size=50, overlap=10)
        assert chunks == []

    def test_process_document_without_optional_fields(self):
        """Test processing document missing optional fields."""
        doc = {
            'content': 'This is some content without optional fields'
        }

        chunks = process_documents([doc], chunk_size=20, overlap=5)

        # Should still create chunks with default values
        assert len(chunks) > 0
        assert chunks[0]['source_doc_id'] == 'unknown'
        assert chunks[0]['source_title'] == 'Untitled'
        assert chunks[0]['category'] == 'general'

    def test_process_cleans_text(self):
        """Test that processing includes text cleaning."""
        doc = {
            'id': 'test',
            'title': 'Test',
            'content': 'Text  with   extra    spaces',
            'category': 'test'
        }

        chunks = process_documents([doc], chunk_size=50, overlap=10)

        # Text should be cleaned (no extra spaces)
        assert '  ' not in chunks[0]['text']

    def test_process_different_chunk_sizes(self, sample_documents):
        """Test processing with different chunk sizes."""
        # Small chunks
        small_chunks = process_documents(sample_documents, chunk_size=20, overlap=5)

        # Large chunks
        large_chunks = process_documents(sample_documents, chunk_size=100, overlap=10)

        # Smaller chunk_size should create more chunks
        assert len(small_chunks) > len(large_chunks)

    def test_process_chunk_text_length(self, sample_documents):
        """Test that chunk text is reasonable length."""
        chunks = process_documents(sample_documents, chunk_size=50, overlap=10)

        for chunk in chunks:
            # Text should not be empty
            assert len(chunk['text']) > 0

            # Text should not be excessively long
            # (allowing some buffer beyond chunk_size)
            word_count = len(chunk['text'].split())
            assert word_count <= 60  # chunk_size + buffer


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_pipeline(self, temp_json_file):
        """Test complete pipeline from file to chunks."""
        # Load
        docs = load_documents(temp_json_file)
        assert len(docs) > 0

        # Process
        chunks = process_documents(docs, chunk_size=50, overlap=10)
        assert len(chunks) > 0

        # Verify structure
        for chunk in chunks:
            assert isinstance(chunk['id'], str)
            assert isinstance(chunk['text'], str)
            assert isinstance(chunk['chunk_index'], int)
            assert isinstance(chunk['total_chunks'], int)

    def test_realistic_document_sizes(self):
        """Test with realistic document sizes."""
        # Create realistic documents
        realistic_docs = [
            {
                'id': 'doc1',
                'title': 'Python Tutorial',
                'content': ' '.join(['word'] * 500),  # ~500 word document
                'category': 'tutorial'
            },
            {
                'id': 'doc2',
                'title': 'ML Guide',
                'content': ' '.join(['word'] * 1000),  # ~1000 word document
                'category': 'guide'
            }
        ]

        chunks = process_documents(realistic_docs, chunk_size=200, overlap=50)

        # Should create reasonable number of chunks
        assert len(chunks) > 5
        assert len(chunks) < 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
