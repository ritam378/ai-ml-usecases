"""
RAG Pipeline - Retrieval-Augmented Generation

Orchestrates the complete RAG workflow:
1. Retrieve relevant chunks
2. Assemble context
3. Generate answer (template or LLM)

Focus: Clear, interview-ready implementation.
"""

from typing import List, Dict, Optional
import json
from document_processor import load_documents, process_documents
from vector_store import VectorStore


class RAGPipeline:
    """
    Complete RAG system for question answering.

    Simple interface for learning and interviews.
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        chunk_size: int = 400,
        overlap: int = 50
    ):
        """
        Initialize RAG pipeline.

        Parameters:
        -----------
        collection_name : str
            Name for vector store collection
        chunk_size : int
            Words per chunk
        overlap : int
            Overlap between chunks
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Initialize vector store
        self.vector_store = VectorStore(collection_name=collection_name)

        print(f"RAG Pipeline initialized")
        print(f"  Collection: {collection_name}")
        print(f"  Chunk size: {chunk_size} words")
        print(f"  Overlap: {overlap} words")

    def add_documents(self, filepath: str):
        """
        Load and index documents from file.

        This builds the knowledge base.

        Parameters:
        -----------
        filepath : str
            Path to JSON file with documents
        """
        print(f"\nLoading documents from {filepath}...")

        # Load documents
        documents = load_documents(filepath)

        # Process into chunks
        chunks = process_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

        # Add to vector store
        self.vector_store.add_documents(chunks)

        print(f"✓ Knowledge base ready with {len(chunks)} chunks")

    def retrieve(self, question: str, top_k: int = 3) -> Dict:
        """
        Retrieve most relevant context for question.

        This is the "Retrieval" stage of RAG.

        Parameters:
        -----------
        question : str
            User question
        top_k : int
            Number of chunks to retrieve

        Returns:
        --------
        retrieval_results : Dict
            Retrieved chunks with distances and metadata
        """
        results = self.vector_store.query(question, top_k=top_k)

        print(f"\nRetrieved {len(results['documents'])} chunks for: '{question}'")
        for i, (doc, dist) in enumerate(zip(results['documents'], results['distances']), 1):
            print(f"  {i}. Distance: {dist:.3f}, Text: {doc[:80]}...")

        return results

    def create_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Assemble prompt with retrieved context.

        This is the "Augmentation" stage of RAG.

        Interview Key Point: Clear structure, explicit instructions,
        and citation requests prevent hallucinations.

        Parameters:
        -----------
        question : str
            User question
        contexts : List[str]
            Retrieved context chunks

        Returns:
        --------
        prompt : str
            Formatted prompt for LLM
        """
        # Join contexts with separators
        context_text = "\n\n---\n\n".join(
            [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)]
        )

        prompt = f"""You are a helpful assistant answering questions based on provided context.

Context:
{context_text}

Instructions:
- Answer the question using ONLY the information in the context above
- If the context doesn't contain enough information, say "I don't have enough information to answer this question"
- Cite which context(s) you used (e.g., "Based on Context 1...")
- Be concise and accurate

Question: {question}

Answer:"""

        return prompt

    def generate_answer_template(self, question: str, contexts: List[str], distances: List[float]) -> Dict:
        """
        Generate answer using template (no LLM needed).

        Good for:
        - Learning the RAG pipeline without API costs
        - Testing retrieval quality
        - Demonstrating the concept

        For actual generation, use generate_answer_llm() instead.

        Parameters:
        -----------
        question : str
            User question
        contexts : List[str]
            Retrieved contexts
        distances : List[float]
            Similarity scores

        Returns:
        --------
        response : Dict
            {
                'answer': str,
                'contexts': List[str],
                'distances': List[float],
                'prompt': str
            }
        """
        # Create prompt (shows what would go to LLM)
        prompt = self.create_prompt(question, contexts)

        # Template answer showing retrieved context
        answer = f"**Retrieved {len(contexts)} relevant contexts:**\n\n"

        for i, (ctx, dist) in enumerate(zip(contexts, distances), 1):
            answer += f"**Context {i}** (similarity: {1-dist:.2f}):\n"
            answer += f"{ctx[:300]}...\n\n"

        answer += f"\n**Question:** {question}\n\n"
        answer += "**Note:** In production, an LLM would generate an answer based on these contexts.\n"
        answer += "For now, review the contexts above to formulate your answer."

        return {
            'answer': answer,
            'contexts': contexts,
            'distances': distances,
            'prompt': prompt
        }

    def query(self, question: str, top_k: int = 3, use_llm: bool = False) -> Dict:
        """
        Answer question using RAG.

        Complete pipeline:
        1. Retrieve relevant chunks
        2. Assemble prompt
        3. Generate answer

        Parameters:
        -----------
        question : str
            User question
        top_k : int
            Number of chunks to retrieve
        use_llm : bool
            Whether to use LLM API (requires setup)

        Returns:
        --------
        response : Dict
            {
                'answer': str,
                'contexts': List[str],
                'distances': List[float],
                'prompt': str
            }
        """
        # Retrieve relevant chunks
        retrieval_results = self.retrieve(question, top_k=top_k)

        contexts = retrieval_results['documents']
        distances = retrieval_results['distances']

        # Generate answer
        if use_llm:
            # Would call LLM API here
            # For learning, we use template approach
            print("\nNote: LLM not configured, using template approach")

        response = self.generate_answer_template(question, contexts, distances)

        return response

    def evaluate_retrieval(self, test_questions_file: str) -> Dict:
        """
        Evaluate retrieval quality using test questions.

        Metrics:
        - Precision@K: Of K retrieved, how many are relevant?
        - Recall@K: Of all relevant, how many retrieved?

        Parameters:
        -----------
        test_questions_file : str
            Path to test questions JSON

        Returns:
        --------
        metrics : Dict
            Evaluation metrics
        """
        # Load test questions
        with open(test_questions_file, 'r') as f:
            test_questions = json.load(f)

        precision_scores = []
        recall_scores = []

        print(f"\nEvaluating on {len(test_questions)} test questions...")

        for item in test_questions:
            question = item['question']
            expected_doc_ids = set(item['expected_doc_ids'])

            # Retrieve
            results = self.retrieve(question, top_k=3)

            # Extract source doc IDs from retrieved chunks
            retrieved_doc_ids = set()
            for metadata in results['metadatas']:
                retrieved_doc_ids.add(metadata['source_doc_id'])

            # Calculate metrics
            relevant_retrieved = retrieved_doc_ids & expected_doc_ids

            if len(results['ids']) > 0:
                precision = len(relevant_retrieved) / len(results['ids'])
            else:
                precision = 0

            if len(expected_doc_ids) > 0:
                recall = len(relevant_retrieved) / len(expected_doc_ids)
            else:
                recall = 0

            precision_scores.append(precision)
            recall_scores.append(recall)

        # Average metrics
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

        metrics = {
            'precision@3': avg_precision,
            'recall@3': avg_recall,
            'num_questions': len(test_questions)
        }

        print(f"\nRetrieval Evaluation Results:")
        print(f"  Precision@3: {avg_precision:.3f}")
        print(f"  Recall@3: {avg_recall:.3f}")
        print(f"  Questions tested: {len(test_questions)}")

        return metrics


if __name__ == '__main__':
    """
    Example usage demonstrating complete RAG workflow.
    """
    print("="*60)
    print("RAG SYSTEM DEMO")
    print("="*60)

    # Initialize pipeline
    rag = RAGPipeline(
        collection_name="demo_kb",
        chunk_size=200,  # Smaller chunks for demo
        overlap=30
    )

    # Add documents
    rag.add_documents('data/knowledge_base.json')

    # Query system
    print("\n" + "="*60)
    print("QUERY EXAMPLE")
    print("="*60)

    question = "What is a Python list?"
    response = rag.query(question, top_k=3)

    print(f"\n{response['answer']}")

    # Evaluate retrieval
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    metrics = rag.evaluate_retrieval('data/test_questions.json')

    print("\n✓ RAG Pipeline demo complete!")
