# RAG System Interview Questions & Answers

Common interview questions about RAG systems with detailed answers. **Study this thoroughly before interviews!**

---

## Quick Reference

**Most Common Questions:**
1. What is RAG and why do we need it?
2. Explain the RAG pipeline step-by-step
3. How do you choose chunk size?
4. How do you evaluate RAG systems?
5. Design a RAG system for [use case]

**Study this file + run the notebooks + review trade_offs.md = Interview ready!**

---

## 1. Foundational Concepts

### Q1: What is RAG and why do we need it?

**Answer:**
RAG = Retrieval-Augmented Generation. It's a technique to improve LLM outputs by retrieving relevant information from an external knowledge base before generation.

**Why we need it:**
- **Reduces hallucinations** - Grounds LLM in real data
- **Up-to-date information** - No need to retrain for new facts
- **Domain-specific knowledge** - Add proprietary data without fine-tuning
- **Cost-effective** - Cheaper than fine-tuning
- **Transparency** - Can cite sources

**Example:** Without RAG, LLM might make up company revenue. With RAG, it retrieves and cites actual financial reports.

---

### Q2: Explain the RAG pipeline step-by-step

**Answer:**

**3 main stages:**

1. **Retrieval** - Find relevant chunks from knowledge base
2. **Augmentation** - Format chunks + question into prompt
3. **Generation** - LLM generates answer using context

**Detailed workflow:**
```
User Query
  → Embed query (convert to vector)
  → Search vector DB (find similar chunks)
  → Retrieve top-k chunks
  → Format prompt (context + question)
  → LLM generates answer
  → Return answer (with citations)
```

---

### Q3: What's the difference between fine-tuning and RAG?

**Answer:**

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Use case | Dynamic facts, documents | Behavior, style, domain reasoning |
| Cost | Low | High (GPU, training) |
| Update speed | Instant (add docs) | Slow (retrain) |
| Best for | Recent data, citations | Specific tasks, tone |

**Many systems use BOTH:** Fine-tune for reasoning, RAG for facts.

---

## 2. System Design

### Q4: Design a RAG system for customer support

**Answer:**

**Key decisions:**
1. **Chunking:** 300 words (support docs are short sections)
2. **Embeddings:** OpenAI ada-002 (quality matters for customers)
3. **Vector DB:** Pinecone (managed, scales well)
4. **Reranking:** Yes (improves top-3 accuracy)
5. **LLM:** GPT-4 (accurate, can cite sources)
6. **Caching:** Redis (30% of questions repeat)

**Architecture:**
```
User Query → Embed → Vector Search (top-5)
→ Rerank (top-3) → Format Prompt → GPT-4
→ Answer + Citations
```

**Estimated cost:** ~$0.035/query

**Follow-up:** "How would you scale to 100K queries/day?"
- Add load balancer, multiple LLM replicas, increase cache

---

## 3. Implementation

### Q5: How do you choose chunk size?

**Answer:**

**Factors:**
- Document structure (blog: 400-600 words, API docs: 100-200)
- Query types (specific facts: smaller, comprehensive: larger)
- LLM context window (GPT-4: can fit 3-4 large chunks)

**My approach:**
1. Start with 400 words (industry default)
2. Create test set (20-50 labeled questions)
3. Evaluate sizes: 200, 300, 400, 600, 800
4. Measure precision/recall for each
5. Pick best F1 score

**Overlap:** Use 10-15% (e.g., 50 words for 400-word chunks)

---

### Q6: How does vector similarity search work?

**Answer:**

**Process:**
1. **Embed text** → Convert to vector (list of numbers)
   - "Python programming" → [0.23, -0.45, 0.67, ..., 0.12]

2. **Measure similarity** → Cosine similarity
   - Similar texts have similar vectors
   - Returns score 0-1 (1 = identical)

3. **Search** → Find nearest vectors to query
   - Naive: Compare to all (slow)
   - Production: Approximate Nearest Neighbor (ANN) - fast

**Why it works:** Embedding models learn that similar meanings → similar vectors

---

## 4. Optimization

### Q7: Your RAG has low precision (60%). How do you fix it?

**Answer:**

**Debug process:**
1. **Collect failure examples** - Where did it go wrong?
2. **Check embedding quality** - Are similarities correct?
3. **Analyze chunk boundaries** - Info split across chunks?
4. **Test query-document mismatch** - Different phrasing?

**Common fixes:**
- **Upgrade embedding model** (MiniLM → mpnet → OpenAI)
- **Semantic chunking** (split at natural boundaries)
- **Query expansion** (generate multiple query variations)
- **Hybrid search** (combine keyword + semantic)
- **Add context to chunks** (prepend doc title)

**Typical improvement:** 60% → 85% precision

---

### Q8: How do you balance cost vs quality?

**Answer:**

**Strategies:**
1. **Tiered LLM** - Simple queries → GPT-3.5, Complex → GPT-4
2. **Adaptive top_k** - Factual queries: k=1, Comparative: k=5
3. **Caching** - Cache common questions (20-40% hit rate)
4. **Batch processing** - For non-real-time (80% cost reduction)

**Cost-quality matrix:**
- Budget: $0.005/query, F1=0.75
- Balanced: $0.035/query, F1=0.85 ← Recommended
- Premium: $0.12/query, F1=0.92

---

## 5. Evaluation

### Q9: How do you evaluate a RAG system?

**Answer:**

**Three metric types:**

**1. Retrieval Metrics**
- **Precision@K:** Of K retrieved, % relevant
- **Recall@K:** Of all relevant, % retrieved
- **F1:** Harmonic mean of precision and recall

**2. Generation Metrics**
- **Accuracy:** Is answer correct?
- **Groundedness:** Sticks to context? (no hallucinations)
- **Relevance:** Addresses the question?

**3. System Metrics**
- **Latency:** p95 < 3 seconds
- **Cost:** $/query
- **Throughput:** queries/second

**Evaluation process:**
1. Create test set (questions + labeled relevant docs)
2. Run RAG pipeline
3. Calculate metrics
4. Compare configurations
5. Choose best trade-off

---

## 6. Production

### Q10: How do you monitor production RAG?

**Answer:**

**Key metrics:**
1. **Quality** - User satisfaction (thumbs up/down), answer acceptance rate
2. **Performance** - p50/p95/p99 latency, error rate
3. **Cost** - Daily spend, cost/query, token usage
4. **System** - Cache hit rate, vector DB size, uptime

**Alerting:**
- CRITICAL: Error rate > 5%, latency > 5s, cost > $0.10/query
- WARNING: Quality < 4.0, cache hit < 20%

**Dashboard:** Show quality, latency, cost, traffic in real-time

---

### Q11: Scale RAG to 1M queries/day?

**Answer:**

**Architecture:**
1. **Horizontal scaling** - Load balancer + multiple RAG servers
2. **Distributed caching** - Redis cluster (40-60% hit rate)
3. **Vector DB** - Managed service (Pinecone) or sharding
4. **Async processing** - Handle 10K concurrent requests
5. **Geographic distribution** - Deploy in multiple regions

**Cost optimization at scale:**
- Caching: Save 40-60%
- Tiered LLM: Save another 30-40%
- Final cost: ~$0.02/query = $20K/day

---

## 7. Advanced

### Q12: What advanced RAG techniques do you know?

**Answer:**

**1. Hybrid Search** - Combine keyword (BM25) + semantic
- Improvement: +10-15% retrieval quality

**2. Reranking** - Get top-20, rerank to top-3 with better model
- Improvement: +15-20% precision, +200ms latency

**3. Query Expansion** - Generate multiple query variations
- Improvement: +10% recall

**4. Hypothetical Document Embeddings (HyDE)** - Generate hypothetical answer, embed that instead of query
- Improvement: +5-10% for complex queries

**5. Parent-Child Retrieval** - Search small chunks (precision), return parent context (completeness)
- Best of both worlds

**Best ROI:** Reranking and hybrid search

---

## Interview Prep Checklist

**Can you explain:**
- ✅ What RAG is and its benefits
- ✅ RAG pipeline stages
- ✅ Vector similarity search
- ✅ Chunking strategies
- ✅ Evaluation metrics
- ✅ Cost optimization
- ✅ Scaling approaches
- ✅ Debugging failures

**Can you design:**
- ✅ A RAG system from scratch
- ✅ An evaluation framework
- ✅ A monitoring system
- ✅ Scaled architecture

**Can you debug:**
- ✅ Low precision
- ✅ Hallucinations
- ✅ High latency
- ✅ High costs

---

## Study Tips

1. **Practice explaining** - Talk through RAG pipeline out loud
2. **Draw diagrams** - Sketch architecture on whiteboard
3. **Know the numbers** - Memorize key metrics (80% precision, 0.81 F1, etc.)
4. **Run the notebooks** - Hands-on experience is crucial
5. **Review trade_offs.md** - Understand design decisions
6. **Prepare examples** - Real scenarios for using RAG

**Most Important:** Understand WHY each decision was made, not just WHAT was implemented.

**Good luck! 🚀**
