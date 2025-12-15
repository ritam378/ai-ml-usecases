# E-commerce Recommendations - Interview Questions & Answers

## Algorithm Questions

### Q1: Explain collaborative filtering vs. content-based filtering

**Answer:**
- **Collaborative Filtering**: Finds patterns in user-product interactions. "Users who liked X also liked Y"
  - Pros: Discovers unexpected patterns, no product features needed
  - Cons: Cold-start for new users/items, popularity bias
  - Example: Netflix recommendations based on viewing patterns

- **Content-Based**: Recommends similar items based on product features
  - Pros: Works for new items, explainable, no need for other users
  - Cons: Filter bubbles, requires good metadata
  - Example: "More like this" based on product description

**Interview Tip**: Always mention when to use each!

### Q2: How do you handle the cold-start problem?

**Answer** (Have 3 solutions ready):
1. **New Users**:
   - Show popular/trending items
   - Use demographic info if available
   - Ask preferences during onboarding
   - Give heavy weight to first interactions

2. **New Items**:
   - Use content-based filtering (works immediately with metadata)
   - Promote to early adopters/power users
   - Bootstrap with editorial picks

3. **New Users + New Items**:
   - Show popular items in same category
   - Category-based recommendations
   - Ask user for preferences

### Q3: What is matrix factorization and why use it?

**Answer:**
- Factorizes user-item matrix R into User×Factors and Item×Factors matrices
- R ≈ U × V^T where k = number of latent factors
- Discovers hidden patterns (e.g., "action movie lovers", "eco-conscious shoppers")
- Handles sparsity better than direct similarity
- More efficient than computing all pairwise similarities

**Trade-off**: Number of factors (k):
- Too small: Underfitting, misses patterns
- Too large: Overfitting, slow training
- Typical range: 50-200

### Q4: How would you build a hybrid recommender?

**Answer** (Describe 3 strategies):

1. **Weighted Combination** (Simplest):
   ```
   score = w1*collab + w2*content + w3*popularity
   ```
   - Active users: Higher w1
   - New users: Higher w3
   - New items: Higher w2

2. **Cascade**:
   - Try collaborative first
   - Fall back to content if insufficient
   - Fall back to popularity as last resort

3. **Feature Augmentation**:
   - Use collaborative features IN content-based model
   - Or vice versa

**Interview Tip**: Explain WHY hybrid solves weaknesses of individual approaches

### Q5: How do you incorporate implicit feedback?

**Answer:**
- Most e-commerce data is implicit (views, clicks, purchases) not explicit (ratings)
- Assign confidence weights:
  ```
  view: 1.0
  add-to-cart: 3.0
  purchase: 5.0
  ```
- Sum for multiple interactions of same user-item pair
- Use algorithms designed for implicit feedback (ALS vs. SVD)

**Discussion Point**: How to set weights? Business-specific, requires experimentation

---

## System Design Questions

### Q6: How do you scale to millions of users?

**Answer:**
1. **Pre-computation**: Batch compute recommendations offline, serve from cache
2. **Approximate NN**: Use FAISS/Annoy for similarity search (O(log n) vs O(n))
3. **Sparse matrices**: Essential for memory efficiency
4. **Distributed training**: Use Spark for large-scale matrix factorization
5. **Tiered strategy**:
   - Active users: Full personalization
   - Occasional users: Cached recommendations
   - New users: Popular items

### Q7: How do you achieve < 100ms latency?

**Answer:**
1. **Pre-compute** recommendations daily/hourly
2. **Multi-level caching**:
   - L1: In-memory (Redis) for popular items
   - L2: Database cache for all users
   - L3: Real-time computation (fallback)
3. **Approximate methods** instead of exact ranking
4. **Limit candidate pool**: Score top-1000 items, not entire catalog

**Trade-off**: Latency vs. freshness

### Q8: How do you monitor recommendation quality?

**Answer:**

**Offline Metrics** (computed on historical data):
- Precision@K, Recall@K, NDCG@K
- Coverage, Diversity
- Fast to compute, but limited

**Online Metrics** (A/B testing):
- Click-through rate (CTR)
- Conversion rate
- Revenue per user
- User engagement (time on site)

**Why online > offline**: Offline doesn't capture user satisfaction, selection bias

---

## Evaluation Questions

### Q9: How do you evaluate recommendations offline?

**Answer:**
1. **Time-based split** (NOT random!):
   ```
   Train: Data before date X
   Test: Data after date X
   ```
2. **Hold-out method**: Hide some user interactions, try to predict them
3. **Metrics**: Precision@K, Recall@K, NDCG@K

**Common Mistake**: Random split causes data leakage!

### Q10: How do you run A/B tests?

**Answer:**
1. **Split users** into control vs. treatment (50/50)
2. **Consistent hashing**: Same user always gets same version
3. **Monitor metrics**: CTR, conversion, revenue
4. **Statistical significance**: Run 2-4 weeks, check p-value < 0.05
5. **Consider all metrics**: Not just accuracy, also business impact

---

## Business Questions

### Q11: How do you balance accuracy with diversity?

**Answer:**
- Pure relevance → filter bubbles
- Pure diversity → irrelevant items
- Solution: **MMR (Maximal Marginal Relevance)**
  ```
  score = λ*relevance + (1-λ)*diversity
  ```
- E-commerce: λ = 0.6-0.8 (favor relevance)
- News/content: λ = 0.4-0.6 (more diversity)

### Q12: How do you explain recommendations?

**Answer:**
Track which component scored highest:
- "Popular among users like you" (collaborative)
- "Similar to [Product X]" (content-based)
- "Trending in [Category]" (popularity)
- "Frequently bought together" (association rules)

**Why important**: Builds user trust, helps debugging

---

## Key Takeaways for Interviews

1. **Start with clarifying questions**: Data, scale, latency, explainability?
2. **Discuss multiple approaches**: Don't jump to one solution
3. **Cold-start is #1 topic**: Have 3 concrete solutions ready
4. **Think about scale**: Pre-computation, caching, approximate methods
5. **Evaluation matters**: Both offline AND online metrics
6. **Be practical**: Balance accuracy, latency, explainability, business needs
7. **Show system thinking**: Not just algorithms, but deployment, monitoring, A/B testing
