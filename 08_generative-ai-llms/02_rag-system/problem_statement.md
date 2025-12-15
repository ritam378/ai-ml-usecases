# Problem Statement: RAG System for Enterprise Knowledge Base

## Business Context

Every large organization has massive amounts of documentation, policies, technical specs, and institutional knowledge scattered across wikis, SharePoint, Confluence, and PDFs. Employees waste hours searching for information, and traditional keyword search often fails to find relevant context.

**The Modern Solution**: Retrieval-Augmented Generation (RAG)
- Combines semantic search with LLM generation
- Answers questions using company's own knowledge
- More accurate than pure LLMs (no hallucinations on company-specific info)
- More flexible than keyword search (understands intent)

## The Core Challenge: Question Answering Over Documents

**Problem**: How do you enable natural language Q&A over a large document corpus?

**Traditional Approaches (Limitations)**:
- **Keyword search**: Misses semantic matches ("profit" vs "revenue")
- **Fine-tuning LLM**: Expensive, static knowledge, doesn't scale
- **Prompt all docs**: Exceeds context limits (most LLMs: 8K-128K tokens)

**RAG Solution**:
1. **Retrieve**: Find most relevant document chunks (semantic search)
2. **Augment**: Add retrieved context to prompt
3. **Generate**: LLM answers using retrieved context

## Real-World Scenario: Technical Support Knowledge Base

Imagine building a RAG system for a software company's support team:

### Requirements
1. **Q&A over 10,000+ support articles**
2. **Semantic search**: "How do I reset password?" finds "password recovery"
3. **Accurate answers**: Cite sources, no hallucinations
4. **Fast retrieval**: <500ms to find relevant docs
5. **Easy updates**: Add new articles without retraining

### Business Impact
- **Support ticket resolution**: 40% → 60% self-service rate
- **Agent productivity**: 30% faster ticket resolution
- **Customer satisfaction**: 24/7 instant answers
- **Cost savings**: $200K/year in reduced support load

## RAG Architecture: The 3-Stage Pipeline

```
User Question: "How do I configure SSL certificates?"
                    ↓
        [1. RETRIEVAL STAGE]
                    ↓
    Document Embedding → Vector Search
                    ↓
    Top 3-5 most relevant chunks found
                    ↓
        [2. AUGMENTATION STAGE]
                    ↓
    Assemble prompt with context:
    "Based on these documents: [chunks]
     Answer: How do I configure SSL certificates?"
                    ↓
        [3. GENERATION STAGE]
                    ↓
    LLM generates answer using context
                    ↓
    "To configure SSL certificates: 1. Navigate to..."
```

## Key Learning Objective

**How do you build a RAG system that retrieves relevant information and generates accurate answers?**

Answer: **Vector embeddings + semantic search + context-aware prompting**

## Dataset Characteristics

For this learning example, we create a small knowledge base:

### Knowledge Base: Python & ML Documentation
- **20-30 documents**: Python basics, ML concepts, data structures
- **Document types**: FAQs, how-to guides, concept explanations
- **Length**: 200-1000 words each
- **Format**: JSON with title, content, metadata

### Test Questions
- **10-15 questions** covering different topics
- **Expected answers** for evaluation
- **Difficulty levels**: Easy (direct match), Hard (requires reasoning)

### Why This Dataset?
1. **Familiar domain**: Python/ML concepts you know
2. **Easy to verify**: Can check if answers are correct
3. **Demonstrates concepts**: Semantic search, chunking, context assembly
4. **Transferable**: Same approach works for any document corpus

## The Three Core Challenges

### Challenge 1: Document Chunking

**Problem**: Documents are long (1000+ words), but LLMs have context limits and retrieval works better on focused chunks.

**Strategy**:
- Split documents into 200-400 word chunks
- Overlap chunks by 20-40 words (preserve context)
- Keep semantic units together (don't split mid-paragraph)

**Trade-offs**:
- **Small chunks** (100 words): Precise but miss context
- **Large chunks** (1000 words): Context-rich but less precise
- **Sweet spot**: 200-400 words with overlap

### Challenge 2: Embedding & Retrieval

**Problem**: How to find semantically similar documents?

**Solution**: Vector embeddings
- Convert text to dense vectors (384 or 768 dimensions)
- Similar meanings → similar vectors
- Use cosine similarity to find matches

**Models**:
- **sentence-transformers/all-MiniLM-L6-v2**: Fast, good quality (384 dims)
- **sentence-transformers/all-mpnet-base-v2**: Better quality (768 dims)

**Vector Store Options**:
- **ChromaDB**: Simple, embedded, perfect for learning
- **FAISS**: Facebook's library, very fast
- **Pinecone/Weaviate**: Managed, production-scale

### Challenge 3: Context Assembly & Prompting

**Problem**: How to combine retrieved chunks into effective prompts?

**Strategy**:
```python
prompt = f"""Answer the question based on the context below.

Context:
{chunk_1}
---
{chunk_2}
---
{chunk_3}

Question: {user_question}

Answer: """
```

**Best Practices**:
- Include 3-5 top chunks (not too many)
- Add clear separators between chunks
- Instruct to cite sources
- Tell model to say "I don't know" if context doesn't help

## Success Metrics

### Retrieval Quality
1. **Precision@K**: Of top-K retrieved docs, how many are relevant?
2. **Recall@K**: Of all relevant docs, how many in top-K?
3. **MRR (Mean Reciprocal Rank)**: Position of first relevant doc

### Generation Quality
1. **Answer Accuracy**: Does answer match ground truth?
2. **Faithfulness**: Is answer supported by retrieved context?
3. **Completeness**: Does answer address full question?

### System Performance
1. **Latency**: Time from question to answer (<1 second ideal)
2. **Cost**: LLM API calls per query

## Interview-Relevant Scenarios

This use case prepares you to discuss:

### 1. RAG vs Fine-Tuning

**Q**: "When would you use RAG vs fine-tuning an LLM?"

**A**: "Use RAG when:
- Knowledge changes frequently (docs updated weekly)
- Need to cite sources (compliance, trust)
- Multiple knowledge domains (different products)
- Cost-sensitive (fine-tuning is expensive)

Use fine-tuning when:
- Need to change model behavior/style
- Knowledge is static and proprietary
- Want lower latency (no retrieval step)"

### 2. Chunking Strategy

**Q**: "How do you decide chunk size?"

**A**: "Balance precision vs context:
- **Test empirically**: Try 100, 200, 400, 800 words
- **Consider domain**: Technical docs → larger chunks (need context), FAQs → smaller chunks (self-contained)
- **Overlap**: 20-40 word overlap preserves context across boundaries
- **Semantic boundaries**: Don't split mid-sentence or mid-thought"

### 3. Embedding Model Selection

**Q**: "Which embedding model should you use?"

**A**: "Depends on requirements:
- **Speed priority**: MiniLM-L6 (fast, 384 dims, good quality)
- **Quality priority**: mpnet-base (slower, 768 dims, better)
- **Domain-specific**: Fine-tune embeddings on your domain
- **Multilingual**: use multilingual models if needed"

### 4. Vector Store Choice

**Q**: "ChromaDB vs Pinecone vs FAISS?"

**A**:
- **ChromaDB**: Simple, embedded, perfect for prototypes and small scale (<1M docs)
- **FAISS**: Fast, local, great for medium scale (1M-10M docs)
- **Pinecone/Weaviate**: Managed, scalable, production-ready (10M+ docs)"

## Learning Objectives

By completing this case study, you will understand:

1. **How RAG works** end-to-end
2. **Vector embeddings** and semantic similarity
3. **Document chunking** strategies
4. **Prompt engineering** for context-aware generation
5. **Evaluation metrics** for retrieval and generation
6. **Common interview questions** about RAG systems

## Simplifications for Learning

To focus on core concepts:
- **Small corpus** (20-30 docs vs thousands)
- **Simple chunking** (fixed size vs semantic)
- **Local vector store** (ChromaDB vs distributed)
- **Optional LLM** (can use templates or mock responses)
- **Single domain** (Python/ML vs multi-domain)

The principles learned apply directly to production RAG systems.

## Common Challenges & Solutions

### Challenge: Hallucination

**Problem**: LLM invents information not in context

**Solutions**:
- Explicit instructions: "Only use provided context"
- Confidence scoring
- Citation requirement
- Temperature = 0 (deterministic)

### Challenge: Retrieval Failures

**Problem**: Wrong documents retrieved

**Solutions**:
- Better chunking strategy
- Hybrid search (semantic + keyword)
- Query rewriting
- Metadata filtering

### Challenge: Context Window Limits

**Problem**: Too many chunks exceed LLM context limit

**Solutions**:
- Retrieve top-3 instead of top-10
- Summarize chunks before assembly
- Use longer-context models (GPT-4-turbo: 128K)
- Re-ranking retrieved chunks

## Next Steps After Mastery

Once comfortable with this implementation:
1. Try different embedding models
2. Implement hybrid search (semantic + BM25)
3. Add metadata filtering (date, category, source)
4. Build evaluation framework (precision, recall, MRR)
5. Add query rewriting/expansion
6. Implement caching for common queries
7. Deploy as REST API
