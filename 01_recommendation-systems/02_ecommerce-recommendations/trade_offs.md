# E-commerce Recommendations - Trade-offs & Design Decisions

## 1. Collaborative vs. Content-Based vs. Hybrid

| Approach | Advantages | Disadvantages | When to Use |
|----------|------------|---------------|-------------|
| **Collaborative Filtering** | - Discovers unexpected patterns<br>- No product features needed<br>- Leverages crowd wisdom | - Cold-start for new users/items<br>- Popularity bias<br>- Sparsity issues | Mature platform with lots of interaction data |
| **Content-Based** | - No cold-start for new items<br>- Explainable<br>- Works with one user | - Filter bubbles<br>- Needs good metadata<br>- Limited discovery | New products frequently added,explainability important |
| **Hybrid** | - Best of both worlds<br>- Handles all cold-start cases<br>- More robust | - More complex<br>- Harder to tune<br>- Higher maintenance | **Most production systems** - balanced approach |

**Our Choice**: Hybrid with context-aware weighting
- Flexibility to adapt based on user activity and context
- Graceful degradation (falls back through collaborative → content → popularity)

---

## 2. Matrix Factorization vs. Deep Learning

| Approach | Pros | Cons | Our Choice |
|----------|------|------|------------|
| **Matrix Factorization** (SVD, ALS, NMF) | - Fast training<br>- Interpretable<br>- Works well with sparse data<br>- Easy to implement | - Limited feature integration<br>- Linear assumptions<br>- Harder to add new features | ✅ **Start here** - Good baseline |
| **Deep Learning** (NCF, Two-Tower, etc.) | - Captures non-linear patterns<br>- Easy to add features<br>- State-of-art performance | - Needs more data<br>- Slower training/inference<br>- Black box<br>- Harder to debug | Future improvement after baseline works |

**Interview Tip**: Always start simple! Deep learning is not always better.

---

## 3. Pre-computation vs. Real-time Computation

| Strategy | Latency | Freshness | Storage | Our Choice |
|----------|---------|-----------|---------|------------|
| **Pre-compute all** | Very low (<10ms) | Stale (24hrs) | High | Homepage recs |
| **Compute on-demand** | High (>500ms) | Real-time | Low | Too slow for production |
| **Hybrid** (pre-compute + adjust) | Low (<100ms) | Semi-fresh (1hr) | Medium | ✅ **Best balance** |

**Our Strategy**:
- Pre-compute base recommendations daily
- Adjust scores for recent activity (last hour)
- Cache results for 1 hour
- Fallback to popular items on cache miss

---

## 4. Accuracy vs. Diversity

| Approach | User Experience | Metrics | Business Impact |
|----------|-----------------|---------|-----------------|
| **Pure Relevance** (λ=1.0) | Filter bubble, boring | High precision | Low exploration, missed cross-sell |
| **Pure Diversity** (λ=0.0) | Random, irrelevant | Low precision | Poor UX, low conversion |
| **Balanced** (λ=0.7) | Relevant + some discovery | Good precision + coverage | ✅ **Best for e-commerce** |

**Tuning λ (MMR parameter)**:
- E-commerce: 0.6-0.8 (favor relevance)
- News/Content: 0.4-0.6 (more diversity)
- Discovery platforms: 0.2-0.4 (high diversity)

---

## 5. Explicit vs. Implicit Feedback

| Feedback Type | Availability | Signal Quality | Handling |
|---------------|--------------|----------------|----------|
| **Explicit** (ratings, thumbs up/down) | Rare (<1% of users) | High quality | Direct use in algorithms |
| **Implicit** (views, clicks, purchases) | Abundant (100% of users) | Noisy | ✅ **Confidence weighting** |

**Our Choice**: Implicit feedback with confidence scores
- `view: 1.0`, `cart: 3.0`, `purchase: 5.0`
- **Rationale**: More data compensates for noise
- **Trade-off**: Need to tune weights (business-specific)

---

## 6. Batch vs. Online Learning

| Strategy | Freshness | Complexity | Resource Usage |
|----------|-----------|------------|----------------|
| **Batch** (retrain daily/weekly) | Moderate | Simple | Scheduled compute |
| **Online** (update after each interaction) | Real-time | Complex | Constant compute |
| **Mini-batch** (retrain hourly) | Fresh | Moderate | ✅ **Good balance** |

**Our Choice**: Daily batch + hourly incremental updates
- Full retrain: Weekly (Sunday night)
- Incremental update: Hourly for active users
- **Why**: Balance freshness with system complexity

---

## 7. Number of Latent Factors (k)

| k value | Model Capacity | Training Speed | Overfitting Risk |
|---------|----------------|----------------|------------------|
| 10-20 | Low | Fast | Low, but may underfit |
| 50-100 | Medium | Medium | ✅ **Good default** |
| 200-500 | High | Slow | High, needs more data |

**Our Choice**: k = 50-100
- **Rationale**: Good balance for mid-size catalog (5K products)
- **Tuning**: Use validation set to find optimal k
- **Rule of thumb**: Start with k = √(num_items)

---

## 8. Similarity Metrics

| Metric | Properties | Use Case |
|--------|------------|----------|
| **Cosine Similarity** | Scale-invariant, [0,1] range | ✅ **Text features, high-dim** |
| **Euclidean Distance** | Absolute distance, sensitive to scale | Normalized numerical features |
| **Pearson Correlation** | Captures linear relationships | Rating patterns |
| **Jaccard** | Set overlap | Categorical features |

**Our Choice**: Cosine similarity
- Works well for TF-IDF features (sparse, high-dimensional)
- Scale-invariant (important when combining different features)
- Interpretable (0 = no similarity, 1 = identical)

---

## 9. Handling Popular Items

| Strategy | Pros | Cons |
|----------|------|------|
| **No adjustment** | Simple | Popularity bias (only recommend bestsellers) |
| **Downweight popular** | Increases diversity | May hurt relevance |
| **Separate popular section** | Clear to users | ✅ **Best UX** |

**Our Approach**: Separate "Trending" section + diversity in personalized recs
- Don't penalize popular items in scoring
- Use MMR to ensure some non-popular items get recommended
- Have explicit "Trending" or "Bestsellers" section

---

## 10. Cold-Start Strategies

| Problem | Solution | Trade-off |
|---------|----------|-----------|
| **New User** | 1. Popular items<br>2. Demographic-based<br>3. Ask preferences | Initial experience vs. onboarding friction |
| **New Item** | 1. Content-based<br>2. Promote to power users<br>3. Editorial picks | Immediate recommendations vs. uncertain quality |
| **New User + New Item** | 1. Category popular<br>2. Ask preferences | Personalization vs. data availability |

**Our Strategy**: Cascade approach
```
IF user_has_history:
    Use collaborative (60%) + content (30%) + popular (10%)
ELIF user_is_new:
    Use popular (60%) + content (30%) + collaborative (10%)
ELSE:  # Cold user + cold item
    Use category-popular (80%) + random-explore (20%)
```

---

## Key Interview Points

1. **No Perfect Solution**: Every choice is a trade-off
2. **Context Matters**: Best approach depends on:
   - Data availability
   - Scale (users, products, interactions)
   - Latency requirements
   - Business objectives

3. **Start Simple**: 
   - Matrix factorization before deep learning
   - Pre-computation before real-time
   - Simple metrics before complex ensembles

4. **Measure Everything**:
   - A/B test all major changes
   - Monitor both offline and online metrics
   - Track business impact, not just model accuracy

5. **Iterate**:
   - Start with baseline (popularity)
   - Add collaborative filtering
   - Add content-based
   - Tune hybrid weights
   - Optimize for scale

**Remember**: The "best" recommendation system is the one that meets business needs while being maintainable!
