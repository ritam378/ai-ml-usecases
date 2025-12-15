"""
Document Processing for RAG System

Handles loading, cleaning, and chunking documents for vector storage.
Focus: Clear, learning-oriented implementation.

Key Learning Points:
- Document chunking strategies
- Text preprocessing
- Metadata preservation
"""

import json
import re
from typing import List, Dict
from pathlib import Path


def load_documents(filepath: str) -> List[Dict]:
    """
    Load documents from JSON file.

    Parameters:
    -----------
    filepath : str
        Path to JSON file with documents

    Returns:
    --------
    documents : List[Dict]
        List of document dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents from {filepath}")
    return documents


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Simple cleaning for learning - in production, might do more.

    Parameters:
    -----------
    text : str
        Raw text

    Returns:
    --------
    cleaned : str
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks.

    Interview Key Point: Chunking balances precision (small chunks)
    vs context (large chunks). Overlap prevents losing information
    at boundaries.

    Parameters:
    -----------
    text : str
        Text to chunk
    chunk_size : int
        Target chunk size in words (~400 is good for most use cases)
    overlap : int
        Number of words to overlap between chunks

    Returns:
    --------
    chunks : List[str]
        List of text chunks
    """
    # Split into words
    words = text.split()

    # If text is shorter than chunk_size, return as single chunk
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        # Get chunk
        end = start + chunk_size
        chunk_words = words[start:end]

        # Join words back into text
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)

        # Move start position (with overlap)
        start += chunk_size - overlap

    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def process_documents(
    documents: List[Dict],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[Dict]:
    """
    Process documents into chunks ready for vector store.

    Steps:
    1. Clean text
    2. Chunk text
    3. Preserve metadata

    Parameters:
    -----------
    documents : List[Dict]
        Raw documents with 'content' and metadata
    chunk_size : int
        Words per chunk
    overlap : int
        Words overlap between chunks

    Returns:
    --------
    chunks : List[Dict]
        Processed chunks with metadata
    """
    all_chunks = []

    for doc in documents:
        # Extract content and metadata
        content = doc.get('content', '')
        doc_id = doc.get('id', 'unknown')
        title = doc.get('title', 'Untitled')
        category = doc.get('category', 'general')

        # Clean content
        content = clean_text(content)

        # Chunk content
        text_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

        # Create chunk documents
        for i, chunk in enumerate(text_chunks):
            chunk_doc = {
                'id': f"{doc_id}_chunk{i}",
                'text': chunk,
                'source_doc_id': doc_id,
                'source_title': title,
                'category': category,
                'chunk_index': i,
                'total_chunks': len(text_chunks)
            }
            all_chunks.append(chunk_doc)

    print(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks


if __name__ == '__main__':
    """
    Example usage and testing.
    """
    # Load documents
    docs = load_documents('data/knowledge_base.json')

    # Process into chunks
    chunks = process_documents(docs, chunk_size=200, overlap=30)

    # Show example chunk
    print("\nExample chunk:")
    print(f"ID: {chunks[0]['id']}")
    print(f"Source: {chunks[0]['source_title']}")
    print(f"Text: {chunks[0]['text'][:200]}...")
    print(f"\nTotal chunks: {len(chunks)}")
