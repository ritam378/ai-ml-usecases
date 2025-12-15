"""
Vector Store for RAG System

Wraps ChromaDB for semantic search with embeddings.
Focus: Simple, clear interface for learning.

Key Learning Points:
- Vector databases and semantic search
- Embedding storage and retrieval
- Similarity metrics
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os


class VectorStore:
    """
    Simple vector store wrapper using ChromaDB.

    ChromaDB handles embedding generation internally using
    sentence-transformers, making it perfect for learning.
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize vector store.

        Parameters:
        -----------
        collection_name : str
            Name for the collection (like a table in SQL)
        persist_directory : str
            Where to store the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Create persistent client
        # This saves embeddings to disk automatically
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create or get collection
        # ChromaDB will use default embedding function (sentence-transformers)
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "RAG knowledge base"}
            )
            print(f"Created new collection: {collection_name}")
        except:
            # Collection already exists
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")

    def add_documents(self, chunks: List[Dict]):
        """
        Add document chunks to vector store.

        ChromaDB automatically generates embeddings for the text.

        Parameters:
        -----------
        chunks : List[Dict]
            Processed chunks with 'id', 'text', and metadata
        """
        if not chunks:
            print("No chunks to add")
            return

        # Extract components
        ids = [chunk['id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]

        # Metadata (optional but useful)
        metadatas = [
            {
                'source_doc_id': chunk.get('source_doc_id', ''),
                'source_title': chunk.get('source_title', ''),
                'category': chunk.get('category', ''),
                'chunk_index': str(chunk.get('chunk_index', 0))
            }
            for chunk in chunks
        ]

        # Add to collection
        # ChromaDB will automatically:
        # 1. Generate embeddings for each document
        # 2. Store embeddings in the vector store
        # 3. Index for fast search
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Added {len(chunks)} chunks to vector store")

    def query(
        self,
        query_text: str,
        top_k: int = 3
    ) -> Dict:
        """
        Search for most similar documents.

        This is the core of RAG: semantic search using embeddings.

        Interview Key Point: ChromaDB embeds the query using the
        same model as documents, then finds closest vectors using
        cosine similarity.

        Parameters:
        -----------
        query_text : str
            Question or query
        top_k : int
            Number of results to return

        Returns:
        --------
        results : Dict
            {
                'documents': List[str],    # Retrieved text chunks
                'distances': List[float],  # Similarity scores (lower=more similar)
                'metadatas': List[Dict],   # Metadata for each chunk
                'ids': List[str]           # Chunk IDs
            }
        """
        # Query collection
        # ChromaDB will:
        # 1. Embed the query text
        # 2. Find top-k most similar document embeddings
        # 3. Return documents, distances, metadata
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )

        # Flatten results (ChromaDB returns list of lists)
        flattened = {
            'documents': results['documents'][0] if results['documents'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }

        return flattened

    def delete_collection(self):
        """Delete the collection (useful for testing)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Could not delete collection: {e}")

    def count(self) -> int:
        """Get number of documents in collection."""
        return self.collection.count()

    def peek(self, limit: int = 5) -> Dict:
        """
        Peek at first few documents in collection.

        Useful for debugging.
        """
        return self.collection.peek(limit=limit)


if __name__ == '__main__':
    """
    Example usage and testing.
    """
    # Create vector store
    store = VectorStore(collection_name="test_kb")

    # Create sample documents
    sample_chunks = [
        {
            'id': 'doc1_chunk0',
            'text': 'Python is a high-level programming language known for its readability.',
            'source_doc_id': 'doc1',
            'source_title': 'Python Intro',
            'category': 'python'
        },
        {
            'id': 'doc2_chunk0',
            'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data.',
            'source_doc_id': 'doc2',
            'source_title': 'ML Basics',
            'category': 'ml'
        }
    ]

    # Add documents
    store.add_documents(sample_chunks)

    # Query
    print("\nQuerying: 'What is Python?'")
    results = store.query("What is Python?", top_k=2)

    print(f"\nTop result:")
    print(f"  Text: {results['documents'][0][:100]}...")
    print(f"  Distance: {results['distances'][0]:.3f}")
    print(f"  Source: {results['metadatas'][0]['source_title']}")

    # Clean up
    store.delete_collection()
