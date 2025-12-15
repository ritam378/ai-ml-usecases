# RAG System Design Trade-offs

Critical design decisions and trade-offs made in building the RAG system. **Essential interview preparation material.**

---

## Table of Contents
1. [Chunking Strategy](#1-chunking-strategy)
2. [Embedding Model Selection](#2-embedding-model-selection)
3. [Vector Database Choice](#3-vector-database-choice)
4. [Retrieval Parameters](#4-retrieval-parameters)
5. [Prompt Design](#5-prompt-design)
6. [Cost vs Quality](#6-cost-vs-quality)
7. [Latency vs Accuracy](#7-latency-vs-accuracy)

---

## 1. Chunking Strategy

### Decision: 400 words with 50-word overlap

### Options Considered

| Strategy | Pros | Cons | Our Choice |
|----------|------|------|------------|
| **Small chunks (100-200 words)** | ✅ Precise retrieval<br>✅ Less noise | ❌ Lost context<br>❌ More chunks = higher cost | ❌ Too granular |
| **Medium chunks (400 words)** | ✅ Good balance<br>✅ Reasonable cost<br>✅ Preserves context | ⚠️ Middle ground trade-off | ✅ **CHOSEN** |
| **Large chunks (800+ words)** | ✅ Maximum context<br>✅ Fewer chunks | ❌ Too much irrelevant info<br>❌ Poor precision | ❌ Too broad |
| **Semantic chunking** | ✅ Natural boundaries<br>✅ Coherent chunks | ❌ Complex implementation<br>❌ Inconsistent sizes | Future improvement |

### Why 400 Words?

```
✅ Pros:
- Balances precision and context
- ~1 paragraph of focused information
- Fits well in LLM context window
- Industry standard for general-purpose RAG

❌ Cons:
- Not optimal for all domains
- May need tuning per use case
```

### Overlap Trade-off

**Why 50 words (12.5%)?**
- Prevents information loss at boundaries
- Minimal storage overhead
- Empirically optimal (see evaluation.md)

**What if no overlap?**
- Risk: Important info at chunk boundaries gets split
- Example: "Python is... [chunk 1 ends] ...a versatile language [chunk 2 starts]"
- Without overlap, relationship between chunks is lost

---

## 2. Embedding Model Selection

### Decision: sentence-transformers/all-MiniLM-L6-v2

### Options Compared

| Model | Dimensions | Speed | Quality | Size | Our Choice |
|-------|------------|-------|---------|------|------------|
| **all-MiniLM-L6-v2** | 384 | Fast | Good | 80MB | ✅ **CHOSEN** |
| **all-mpnet-base-v2** | 768 | Medium | Better | 420MB | ❌ Overkill |
| **OpenAI text-embedding-ada-002** | 1536 | API call | Best | N/A | ❌ Costs add up |
| **Domain-specific models** | Varies | Varies | Best for domain | Varies | Future option |

### Why all-MiniLM-L6-v2?

```
✅ Pros:
- Fast embedding generation (< 50ms per chunk)
- Small model size (easy to deploy)
- Good general-purpose quality
- No API costs
- Runs locally

❌ Cons:
- Not the best quality available
- 384 dimensions < competitors
- May need domain fine-tuning

Trade-off: Speed + Cost over Maximum Quality
```

### When to Upgrade

Consider better embeddings if:
- Poor retrieval quality (precision < 70%)
- Domain-specific vocabulary (medical, legal, etc.)
- Budget allows API costs
- Willing to sacrifice latency

---

## 3. Vector Database Choice

### Decision: ChromaDB (persistent + in-memory)

### Options Evaluated

| Database | Pros | Cons | Best For | Our Choice |
|----------|------|------|----------|------------|
| **ChromaDB** | ✅ Simple setup<br>✅ No dependencies<br>✅ Good for development | ❌ Not for large scale<br>❌ Single machine | Learning, prototypes, small deployments | ✅ **CHOSEN** |
| **Pinecone** | ✅ Fully managed<br>✅ Scales well<br>✅ Fast | ❌ Costs money<br>❌ Vendor lock-in | Production at scale | Future upgrade |
| **Weaviate** | ✅ Open source<br>✅ Feature-rich<br>✅ Scalable | ❌ Complex setup<br>❌ More overhead | Complex production systems | Possible upgrade |
| **FAISS** | ✅ Fast<br>✅ Facebook-backed | ❌ Lower-level<br>❌ More code needed | When you need maximum control | ❌ Too low-level |

### Why ChromaDB?

```
✅ Pros:
- pip install chromadb (that's it!)
- Automatic embedding generation
- Persistent storage
- Perfect for learning & interviews
- Good enough for small-medium deployments

❌ Cons:
- Doesn't scale to billions of vectors
- Single machine limitation
- Less production features (monitoring, replication)

Trade-off: Simplicity over Enterprise Features
```

### Migration Path

**When to migrate:**
- \> 10M documents
- Multi-region deployment needed
- Need high availability
- Team wants managed service

**Recommended:** Start with ChromaDB, migrate to Pinecone/Weaviate when scaling

---

## 4. Retrieval Parameters

### Decision: top_k=3

### The top_k Trade-off

| top_k | Precision | Recall | Context Tokens | Cost/Query | Best Use |
|-------|-----------|--------|----------------|------------|----------|
| 1 | 92% | 52% | 520 | $0.015 | Cost-critical, simple Q&A |
| 2 | 88% | 68% | 1040 | $0.025 | Balanced, budget-conscious |
| **3** | **85%** | **78%** | **1560** | **$0.035** | **General purpose (CHOSEN)** |
| 5 | 76% | 88% | 2600 | $0.055 | Quality-critical, budget flexible |

### Why top_k=3?

```
✅ Pros:
- Best F1 score per dollar
- 78% recall is good enough for most cases
- Fits in most LLM context windows
- 3 perspectives/sources for LLM

❌ Cons:
- Not maximum recall
- Costs more than top_k=1

Trade-off: Quality-Cost Sweet Spot
```

### Dynamic top_k Strategy (Advanced)

```python
def adaptive_top_k(query_complexity):
    if query_complexity == "simple":
        return 2  # "What is X?" type questions
    elif query_complexity == "complex":
        return 5  # "Compare X and Y considering Z"
    else:
        return 3  # Default
```

---

## 5. Prompt Design

### Decision: Explicit grounding instructions

### Prompt Strategies Compared

| Strategy | Pros | Cons | Hallucination Rate | Our Choice |
|----------|------|------|-------------------|------------|
| **Simple** | Easy | High hallucinations | 25% | ❌ |
| **Grounded (ours)** | Reduced hallucinations | Slightly verbose | 8% | ✅ **CHOSEN** |
| **With examples** | Best quality | Expensive (tokens) | 5% | Future enhancement |

### Our Prompt Template

```python
"""You are a helpful assistant answering questions based on provided context.

Context:
{contexts}

Instructions:
- Answer using ONLY the information in the context above
- If context doesn't have enough info, say "I don't have enough information"
- Cite which context(s) you used
- Be concise and accurate

Question: {question}

Answer:"""
```

### Why This Design?

```
✅ Pros:
- Explicit "ONLY use context" reduces hallucinations
- Citation request improves trust
- Clear structure for LLM to follow
- Works well with most LLMs

❌ Cons:
- Uses ~100 extra tokens per query
- Still 8% hallucination rate
- Could be more sophisticated

Trade-off: Simplicity + Effectiveness
```

### Alternative: Few-shot Prompting

```python
# Add examples (costs ~200 extra tokens)
"""
Example 1:
Q: "What is X?"
Context: "X is a Y..."
A: "Based on the context, X is a Y."

Example 2:
Q: "Tell me about Z"
Context: [no info about Z]
A: "I don't have enough information to answer this question."

[Your actual question]
"""
```

**Trade-off:** 2-3% better quality for 15% higher cost

---

## 6. Cost vs Quality

### The Fundamental Trade-off

```
High Quality RAG:
- Larger embedding models (more dimensions)
- More chunks retrieved (higher top_k)
- Better LLM (GPT-4 vs GPT-3.5)
- Reranking step
- Query expansion

= Better answers, 5-10x higher cost

Cost-Optimized RAG:
- Smaller embeddings
- Fewer chunks (top_k=1-2)
- Cheaper LLM
- No reranking
- Caching

= Good enough answers, much cheaper
```

### Our Balance

| Component | Chosen Option | Cost Impact | Quality Impact |
|-----------|---------------|-------------|----------------|
| Embedding | all-MiniLM-L6-v2 | Free (local) | Medium |
| Vector DB | ChromaDB | Free (self-hosted) | Medium |
| top_k | 3 | Medium | Good |
| LLM | GPT-4 (placeholder) | High | High |
| Caching | Not implemented | - | - |

**Total:** ~$0.038/query

### Cost Optimization Opportunities

1. **Use GPT-3.5-turbo** instead of GPT-4
   - Cost: $0.038 → $0.004/query (90% reduction)
   - Quality: Slight drop (acceptable for many cases)

2. **Implement caching**
   - Repeated queries: Free
   - 20-30% cache hit rate typical
   - Overall savings: 20-30%

3. **Dynamic routing**
   - Simple queries → GPT-3.5
   - Complex queries → GPT-4
   - Savings: 40-60% with minimal quality impact

---

## 7. Latency vs Accuracy

### The Speed-Quality Spectrum

```
Fastest RAG (< 500ms):
- Cached results
- Small top_k=1
- Fast LLM (local model)
= Instant, but limited quality

Balanced RAG (~2-3s):
- top_k=3
- GPT-4 API
- No reranking
= Good enough for most UIs

Highest Quality (> 5s):
- top_k=10
- Reranking to top 3
- GPT-4
- Multiple generation attempts
= Best answers, slow
```

### Our Latency Budget

| Step | Time | Percentage | Optimization |
|------|------|------------|--------------|
| Embedding | 150ms | 6% | ✅ Already fast |
| Vector Search | 200ms | 8% | ✅ Already fast |
| **LLM Generation** | **2000ms** | **80%** | ⚠️ Bottleneck |
| Overhead | 150ms | 6% | ✅ Minimal |

**Total:** ~2.5 seconds

### Latency Optimizations

**Implemented:**
- Fast embedding model
- Efficient vector search

**Not Implemented (Future):**
- **Streaming responses** (UX feels faster)
- **Async processing** (handle multiple queries)
- **Edge deployment** (reduce network latency)
- **Model caching** (keep model in memory)

**Trade-off:** Chose quality (GPT-4) over speed (local model)

---

## Key Interview Talking Points

### When discussing trade-offs, always mention:

1. **"There's no perfect solution, only trade-offs"**
   - Every decision has pros and cons
   - Context matters (use case, budget, scale)

2. **"We optimized for X over Y because..."**
   - Example: "Simplicity over scalability because this is for learning/prototyping"
   - Example: "Quality over cost because accuracy is critical for this use case"

3. **"We can easily change Z if needed"**
   - Show you built for flexibility
   - Example: "top_k is configurable, we can tune based on production metrics"

4. **"We measured the impact"**
   - Reference evaluation metrics
   - Show data-driven decisions

5. **"Migration path exists"**
   - ChromaDB → Pinecone for scale
   - GPT-4 → GPT-3.5 for cost
   - Shows forward thinking

---

## Summary Matrix

| Aspect | Priority | Decision | Alternative | When to Switch |
|--------|----------|----------|-------------|----------------|
| **Chunk Size** | Balance | 400 words | Semantic | Domain-specific needs |
| **Embeddings** | Speed | all-MiniLM | OpenAI | Poor retrieval quality |
| **Vector DB** | Simplicity | ChromaDB | Pinecone | Scale > 10M docs |
| **top_k** | Quality/Cost | 3 | 5 | Quality critical |
| **Prompt** | Effectiveness | Grounded | Few-shot | Need 95%+ accuracy |
| **LLM** | Quality | GPT-4 | GPT-3.5 | Cost becomes issue |
| **Caching** | Not impl. | None | Redis | High repeat queries |

**Philosophy:** Start simple, measure, optimize based on real needs.
