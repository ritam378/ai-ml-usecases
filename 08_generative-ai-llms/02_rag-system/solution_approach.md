# Solution Approach: RAG System for Enterprise Knowledge Base

## Overview

This solution demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline using vector embeddings for semantic search and prompt engineering for accurate answer generation. Focus: Clear, interview-ready implementation.

## Architecture

```
Knowledge Base Documents
        ↓
[Document Processing]
    - Load documents
    - Clean text
    - Split into chunks (200-400 words)
        ↓
[Embedding Generation]
    - sentence-transformers model
    - Convert chunks to vectors (384 dims)
        ↓
[Vector Store (ChromaDB)]
    - Store embeddings
    - Index for fast search
        ↓
User Query → Embed Query → Semantic Search → Top-K Chunks
        ↓
[Context Assembly]
    - Combine top chunks
    - Format prompt
        ↓
[LLM Generation]
    - Generate answer from context
    - (Optional: use template if no LLM)
        ↓
Answer + Sources
```

## Key Design Decisions

### 1. Document Chunking Strategy

**Why Chunk Documents?**

**Problem**: Documents are too long for:
- Efficient retrieval (specific info gets diluted)
- LLM context windows (most models: 4K-8K tokens)

**Solution**: Fixed-size chunking with overlap

```python
def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into overlapping chunks.

    chunk_size: ~400 words (optimal for semantic search)
    overlap: ~50 words (preserves context at boundaries)
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
```

**Interview Insight**: "I chose 400-word chunks because:
- Small enough for precise retrieval
- Large enough to preserve context
- 50-word overlap ensures we don't lose information at chunk boundaries
- Can fit 3-5 chunks in typical LLM prompts"

**Alternatives**:
- **Semantic chunking**: Split at paragraph/section boundaries (better but complex)
- **Recursive chunking**: Try large chunks, split if needed
- **Sentence-based**: Use sentence boundaries (simpler but may be too small)

### 2. Embedding Model: sentence-transformers

**Choice**: `all-MiniLM-L6-v2`

**Why This Model?**

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | Fast | Good | Learning, prototypes |
| all-mpnet-base-v2 | 768 | Slower | Better | Production |
| text-embedding-ada-002 | 1536 | API call | Best | OpenAI users |

**For learning, MiniLM is perfect**:
- Fast encoding (~1000 sentences/sec)
- Small model (~80MB)
- Good quality for most tasks
- Works offline

**How Embeddings Work**:

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text to vector
text = "How do I reset my password?"
embedding = model.encode(text)  # Returns: array of 384 floats

# Similar texts have similar vectors (high cosine similarity)
query_vec = model.encode("password reset help")
doc_vec = model.encode("To reset password, go to settings...")
similarity = cosine_similarity(query_vec, doc_vec)  # ~0.85 (very similar)
```

**Interview Key Point**: "Embeddings convert text to dense vectors where semantic similarity is captured by vector proximity. Unlike keyword search, 'password reset' and 'forgot password' will have high similarity even with no word overlap."

### 3. Vector Store: ChromaDB

**Why ChromaDB?**

**Advantages**:
- **Embedded**: No separate server needed
- **Simple API**: Add, query, persist in 3 lines
- **Automatic**: Handles embedding storage, indexing
- **Persistent**: Saves to disk automatically

**Basic Usage**:

```python
import chromadb

# Create client and collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("knowledge_base")

# Add documents (ChromaDB handles embedding internally)
collection.add(
    documents=["Python is a programming language", "Machine learning uses data"],
    ids=["doc1", "doc2"]
)

# Query (returns most similar documents)
results = collection.query(
    query_texts=["What is Python?"],
    n_results=3
)
```

**Interview Comparison**:

**Q**: "Why ChromaDB over FAISS or Pinecone?"

**A**: "For learning and small-scale apps:
- **ChromaDB**: Easiest, embedded, perfect for <100K docs
- **FAISS**: Faster search, local, good for 100K-10M docs, requires more setup
- **Pinecone**: Managed service, scalable, costs money, overkill for learning

I'd start with ChromaDB, migrate to FAISS if performance becomes an issue, and only use Pinecone for production multi-million document scale."

### 4. Retrieval Strategy

**Semantic Search Flow**:

```python
def retrieve_context(query: str, top_k: int = 3):
    """
    Retrieve most relevant chunks for query.

    Steps:
    1. Embed query using same model as documents
    2. Find top-k most similar embeddings (cosine similarity)
    3. Return corresponding text chunks
    """
    # Query vector store
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    # Extract chunks
    chunks = results['documents'][0]
    distances = results['distances'][0]  # Lower = more similar

    return chunks, distances
```

**top_k Selection**:
- **k=1**: Fast but may miss context
- **k=3**: Good balance (recommended)
- **k=5**: More context but redundancy risk
- **k=10**: Usually too much, exceeds context limits

**Interview Insight**: "I use k=3 because:
- Provides enough context for most questions
- Fits comfortably in LLM prompts (~1200 words)
- Low latency (3 similarity computations)
- Higher k shows diminishing returns in quality"

### 5. Prompt Engineering

**Context Assembly**:

```python
def create_prompt(query: str, contexts: list[str]) -> str:
    """
    Assemble prompt with retrieved context.

    Best practices:
    - Clear separation between contexts
    - Explicit instructions
    - Request citations
    """
    # Join contexts with separators
    context_text = "\n---\n".join(contexts)

    prompt = f"""You are a helpful assistant answering questions based on provided context.

Context:
{context_text}

Instructions:
- Answer the question using ONLY the context above
- If the context doesn't contain enough information, say "I don't have enough information"
- Cite which context section you used (Context 1, Context 2, etc.)

Question: {query}

Answer:"""

    return prompt
```

**Prompt Design Principles**:

1. **Clear Role**: "You are a helpful assistant..."
2. **Explicit Constraints**: "using ONLY the context above"
3. **Failure Mode**: Tell model what to do when uncertain
4. **Citation Request**: Improves trust and debugging
5. **Structure**: Separate context, instructions, question

**Interview Q**: "How do you prevent hallucinations?"

**A**: "Multiple strategies:
- Explicit instruction to use only provided context
- Temperature=0 for deterministic output
- Request citations (model less likely to make up sources)
- Post-processing to verify answer comes from context
- Confidence scoring"

### 6. Generation (Optional LLM Integration)

**Two Approaches**:

#### Approach A: Template-Based (No LLM)

For learning without API costs:

```python
def generate_answer_template(query: str, contexts: list[str]) -> str:
    """
    Simple template-based answer (no LLM needed).

    Good for:
    - Learning the RAG pipeline
    - Testing retrieval quality
    - No API costs
    """
    answer = f"Based on the retrieved context:\n\n"

    for i, context in enumerate(contexts, 1):
        answer += f"Context {i}: {context[:200]}...\n\n"

    answer += f"\nQuestion: {query}\n"
    answer += "Answer: [Use LLM or provide manual answer based on context]"

    return answer
```

#### Approach B: LLM API (Production)

For actual generation:

```python
def generate_answer_llm(prompt: str) -> str:
    """
    Generate answer using LLM API.

    Options:
    - OpenAI (gpt-3.5-turbo, gpt-4)
    - Anthropic (claude)
    - Local (llama, mistral via ollama)
    """
    # Example with OpenAI (requires API key)
    import openai

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0  # Deterministic
    )

    return response.choices[0].message.content
```

**For learning**: Start with template approach, add LLM later if desired.

### 7. Evaluation Metrics

**Retrieval Evaluation**:

```python
def evaluate_retrieval(questions, ground_truth_docs):
    """
    Measure retrieval quality.

    Metrics:
    - Precision@K: Of K retrieved docs, how many relevant?
    - Recall@K: Of all relevant docs, how many retrieved?
    - MRR: Position of first relevant doc
    """
    precision_scores = []
    recall_scores = []
    mrr_scores = []

    for q, truth_docs in zip(questions, ground_truth_docs):
        retrieved = retrieve_context(q, top_k=3)

        # Calculate metrics
        relevant_retrieved = set(retrieved) & set(truth_docs)

        precision = len(relevant_retrieved) / len(retrieved)
        recall = len(relevant_retrieved) / len(truth_docs)

        # MRR: 1/rank of first relevant doc
        for rank, doc in enumerate(retrieved, 1):
            if doc in truth_docs:
                mrr = 1 / rank
                break
        else:
            mrr = 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)

    return {
        'precision@3': np.mean(precision_scores),
        'recall@3': np.mean(recall_scores),
        'mrr': np.mean(mrr_scores)
    }
```

**Generation Evaluation** (with LLM):
- **Accuracy**: Does answer match expected answer?
- **Faithfulness**: Is answer supported by retrieved context?
- **Completeness**: Does answer address full question?

## Implementation Details

### Module Organization

```
src/
├── document_processor.py    # Load, clean, chunk documents
├── embedding_utils.py        # Embedding generation with sentence-transformers
├── vector_store.py           # ChromaDB wrapper
├── rag_pipeline.py           # Orchestrate retrieve → generate
└── prompts.py               # Prompt templates
```

**Why This Structure**: Each module has single responsibility, easy to test and modify.

### Class Design

```python
class RAGPipeline:
    """
    Main RAG system orchestrator.

    Simple interface for learning.
    """

    def __init__(self, collection_name: str = "knowledge_base"):
        """Initialize with vector store and embedding model."""

    def add_documents(self, documents: list[dict]):
        """Add documents to knowledge base."""

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Answer question using RAG.

        Returns:
        {
            'answer': str,
            'contexts': list[str],
            'distances': list[float]
        }
        """

    def evaluate(self, test_questions: list[dict]) -> dict:
        """Evaluate retrieval quality."""
```

## Expected Results

With our knowledge base (20-30 Python/ML docs):

**Retrieval Quality**:
- Precision@3: 0.80-0.90 (most retrieved chunks are relevant)
- Recall@3: 0.60-0.70 (finds most relevant information)
- MRR: 0.75-0.85 (relevant doc usually in top 2)

**System Performance**:
- Indexing: <1 second for 30 documents
- Query time: 50-100ms for retrieval
- Total latency: 200ms (retrieval) + LLM time if used

**Example**:

```
Question: "What is a Python list?"

Retrieved Contexts:
1. "Python lists are ordered, mutable collections... [distance: 0.23]"
2. "Lists in Python can contain mixed data types... [distance: 0.31]"
3. "Common list operations include append, extend... [distance: 0.38]"

Answer: "A Python list is an ordered, mutable collection that can
contain mixed data types. You can create lists using square brackets..."
[Based on Context 1, 2]
```

## Code Workflow

### 1. Build Knowledge Base

```python
from rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()

# Load documents
documents = load_json('data/knowledge_base.json')

# Add to vector store (automatic embedding)
rag.add_documents(documents)
```

### 2. Query System

```python
# Ask question
result = rag.query("How do I use Python dictionaries?")

print(result['answer'])
print(f"Sources: {len(result['contexts'])} contexts used")
```

### 3. Evaluate

```python
# Load test questions
test_qa = load_json('data/test_questions.json')

# Evaluate retrieval
metrics = rag.evaluate(test_qa)
print(f"Precision@3: {metrics['precision@3']:.2f}")
```

## Common Interview Questions & Answers

### Q: How does RAG differ from fine-tuning?

**A**: "RAG retrieves relevant information at query time and uses it as context, while fine-tuning bakes knowledge into model weights. RAG advantages:
- Dynamic knowledge (update documents anytime)
- Cheaper (no retraining)
- Citable sources (see where answer comes from)
- Better for factual Q&A

Fine-tuning advantages:
- Changes model behavior/style
- No retrieval latency
- Can work without external knowledge"

### Q: What's the biggest challenge in RAG?

**A**: "Retrieval quality. If you retrieve wrong chunks, the LLM can't generate correct answers. Solutions:
- Better chunking (semantic vs fixed-size)
- Hybrid search (combine semantic + keyword)
- Query rewriting (rephrase for better retrieval)
- Re-ranking (score chunks by relevance)
- Metadata filtering (narrow search space)"

### Q: How do you handle multi-hop questions?

**A**: "Questions requiring multiple pieces of information:
1. **Decompose**: Break into sub-questions
2. **Iterative retrieval**: Retrieve for each sub-question
3. **Chain reasoning**: Use first answer to inform second query
4. **Aggregate**: Combine information from multiple chunks

Example: 'Who invented the programming language Python was written in?'
→ Sub-Q1: 'What language is Python written in?' (C)
→ Sub-Q2: 'Who invented C?' (Dennis Ritchie)"

### Q: How would you deploy this?

**A**: "Progressive deployment:
1. **Prototype**: Local ChromaDB, sentence-transformers
2. **MVP**: Dockerize, add REST API (FastAPI)
3. **Production**:
   - Vector store: Migrate to Pinecone/Weaviate
   - LLM: Managed API (OpenAI) or self-hosted (vLLM)
   - Caching: Redis for common queries
   - Monitoring: Log queries, latency, user feedback
   - A/B testing: Compare retrieval strategies"

## Limitations & Future Improvements

**Current Limitations**:
- Small knowledge base (30 docs)
- Simple chunking (fixed size)
- No query rewriting
- No re-ranking
- Optional LLM (template fallback)

**Production Enhancements**:
- Semantic chunking (split at logical boundaries)
- Hybrid search (BM25 + semantic)
- Query expansion/rewriting
- Re-ranking with cross-encoder
- Metadata filtering (date, source, category)
- Caching popular queries
- User feedback loop
- Multi-modal RAG (images, tables)

## Key Takeaways for Interviews

1. **RAG = Retrieve + Augment + Generate** - Three-stage pipeline
2. **Embeddings capture semantics** - Similar meaning → similar vectors
3. **Chunking is critical** - Balance precision vs context
4. **Prompt engineering matters** - Clear instructions prevent hallucinations
5. **Evaluation has two parts** - Retrieval quality + generation quality
6. **Vector stores enable semantic search** - ChromaDB for learning, scale up as needed
7. **Template approach for learning** - Can demonstrate RAG without LLM API costs
