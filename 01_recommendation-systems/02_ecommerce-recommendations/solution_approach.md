# E-commerce Recommendations - Solution Approach

## Solution Overview

We'll implement a **hybrid recommendation system** that combines three complementary approaches:

1. **Collaborative Filtering**: Find products liked by similar users
2. **Content-Based Filtering**: Find products similar to what the user has liked
3. **Popularity-Based**: Handle cold-start with trending items

This hybrid approach provides:
- **Personalization** from collaborative filtering
- **Explainability** from content-based filtering
- **Robustness** from the hybrid combination
- **Cold-start handling** from popularity fallback

**Interview Key Point**: Always explain WHY you're choosing a hybrid approach - it handles the weaknesses of individual methods.

---

## Architecture Design

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Recommendation System                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Hybrid Recommendation Engine       │
        └─────────────────────────────────────────┘
                    │           │           │
        ┌───────────┴───┬───────┴───┬───────┴────────┐
        │               │           │                │
        ▼               ▼           ▼                ▼
┌──────────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐
│Collaborative │  │Content   │  │Popular │  │   Caching    │
│   Filtering  │  │  Based   │  │  Items │  │    Layer     │
└──────────────┘  └──────────┘  └────────┘  └──────────────┘
        │               │           │
        ▼               ▼           ▼
┌──────────────────────────────────────────────┐
│         Data Layer (Interactions + Metadata)  │
└──────────────────────────────────────────────┘
```

### Component Breakdown

1. **Data Layer**
   - User-product interactions (views, purchases, ratings)
   - Product metadata (category, brand, price, description)
   - Pre-computed similarity matrices

2. **Algorithm Layer**
   - Collaborative filter (matrix factorization)
   - Content-based recommender (TF-IDF + cosine similarity)
   - Popularity ranker

3. **Hybrid Combiner**
   - Weighted scoring
   - Context-aware switching
   - Diversity injection

4. **Serving Layer**
   - Pre-computed recommendations (batch)
   - Real-time score adjustment
   - Caching for low latency

---

## Approach 1: Collaborative Filtering

### Overview

Collaborative filtering finds patterns in user-product interactions: "Users who liked X also liked Y."

**Two Main Types**:
1. **User-based**: Find similar users, recommend what they liked
2. **Item-based**: Find similar items based on co-interaction patterns

**Interview Choice**: For e-commerce, **item-based CF** typically works better because:
- Products are more stable than user preferences
- More interpretable ("bought together")
- Better for cold-start users (not cold-start items)

### Matrix Factorization Approach

Instead of computing user or item similarities directly, we factor the user-item interaction matrix:

```
R (users × items) ≈ U (users × k) × V^T (k × items)
```

Where:
- `R` = sparse user-item interaction matrix
- `U` = user latent feature matrix
- `V` = item latent feature matrix
- `k` = number of latent factors (hyperparameter, typically 50-200)

**Why This Works**:
- Discovers hidden patterns (latent factors)
- Handles sparsity better than direct similarity
- Computationally efficient
- Generalizes well

**Algorithms**:
- **SVD** (Singular Value Decomposition): Classic approach
- **ALS** (Alternating Least Squares): Better for implicit feedback
- **NMF** (Non-Negative Matrix Factorization): Non-negative factors

**For E-commerce**: We'll use **ALS** because most data is implicit (views, purchases without ratings).

### Handling Implicit Feedback

**Challenge**: We have views, clicks, purchases - not explicit ratings.

**Solution**: Create confidence scores based on interaction type:

```python
confidence_scores = {
    'view': 1.0,
    'cart': 3.0,
    'purchase': 5.0,
    'rating': rating_value
}
```

**Interview Discussion Point**: How to weight different signals? This is business-specific:
- Purchases > Cart > Views
- Recent interactions weighted higher (time decay)
- Multiple views of same product = higher confidence

### Cold-Start for New Users

**Problem**: No interaction history for new users.

**Solutions** (discuss multiple in interview):

1. **Popularity-Based**: Show trending items
2. **Demographic-Based**: Group by age/location, use group preferences
3. **Onboarding Questions**: Ask for preferences during signup
4. **Fast Learning**: Give heavy weight to first few interactions

---

## Approach 2: Content-Based Filtering

### Overview

Content-based filtering recommends items similar to what the user has liked, based on item features.

**Example**: If user bought "Wireless Headphones," recommend similar headphones based on:
- Same category
- Similar brands
- Similar price range
- Similar product descriptions

**Advantages**:
- **Works for new items** (cold-start items)
- **Explainable**: "Recommended because similar to X"
- **No need for other users' data**
- **Transparency and user trust**

**Disadvantages**:
- Can create filter bubbles (only similar items)
- Limited serendipity/discovery
- Requires good product metadata

### Feature Engineering

**Text Features** (from product descriptions):
```python
# Use TF-IDF (Term Frequency-Inverse Document Frequency)
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert product descriptions to feature vectors
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
text_features = tfidf.fit_transform(product_descriptions)
```

**Categorical Features**:
```python
# One-hot encode categories
category_features = pd.get_dummies(products['category'])

# Brand features
brand_features = pd.get_dummies(products['brand'])
```

**Numerical Features**:
```python
# Normalize price
from sklearn.preprocessing import StandardScaler
price_features = StandardScaler().fit_transform(products[['price']])
```

**Combined Feature Vector**:
```python
# Concatenate all features
product_features = np.hstack([
    text_features.toarray(),
    category_features.values,
    brand_features.values,
    price_features
])
```

### Similarity Computation

**Cosine Similarity** (most common for recommendations):

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute item-item similarity matrix
item_similarity = cosine_similarity(product_features)
```

**Why Cosine Similarity**:
- Scale-invariant (doesn't matter if features have different ranges)
- Efficient to compute
- Works well for sparse high-dimensional data
- Interpretable (0 = no similarity, 1 = identical)

**Alternative Similarity Metrics**:
- Euclidean distance (sensitive to scale)
- Jaccard similarity (for categorical features)
- Pearson correlation (for rating patterns)

### Cold-Start for New Items

**Advantage**: Content-based filtering naturally handles new items!

As soon as we have product metadata, we can:
1. Compute feature vector
2. Find similar existing products
3. Recommend to users who liked those similar products

---

## Approach 3: Hybrid Recommendation Strategy

### Why Hybrid?

**Interview Key Point**: Always articulate the weaknesses each approach solves:

| Approach | Strengths | Weaknesses |
|----------|-----------|-----------|
| Collaborative | Discovers unexpected patterns, uses wisdom of crowd | Cold-start (users & items), popularity bias |
| Content-Based | Works for new items, explainable | Filter bubbles, needs good metadata |
| Popularity | Simple, works for everyone | No personalization, favors popular items |

Hybrid = Best of all worlds!

### Hybrid Strategies

**1. Weighted Combination** (Simplest - Start Here in Interview)

```python
final_score = (
    w1 * collaborative_score +
    w2 * content_based_score +
    w3 * popularity_score
)
```

**How to set weights** (interview discussion):
- **Active users** (lots of history): Higher w1 (collaborative)
- **New users**: Higher w3 (popularity)
- **New items**: Higher w2 (content-based)

Example:
```python
def get_hybrid_weights(user):
    num_interactions = count_user_interactions(user)

    if num_interactions == 0:  # Brand new user
        return {'collaborative': 0.0, 'content': 0.3, 'popularity': 0.7}
    elif num_interactions < 5:  # New user
        return {'collaborative': 0.2, 'content': 0.3, 'popularity': 0.5}
    else:  # Active user
        return {'collaborative': 0.6, 'content': 0.3, 'popularity': 0.1}
```

**2. Context-Aware Switching**

Different contexts need different approaches:

```python
if context == 'homepage':
    # Personalized recommendations
    weights = {'collaborative': 0.6, 'content': 0.3, 'popularity': 0.1}

elif context == 'product_page':
    # Similar products
    weights = {'collaborative': 0.2, 'content': 0.7, 'popularity': 0.1}

elif context == 'cart':
    # Complementary products (bought together)
    weights = {'collaborative': 0.8, 'content': 0.1, 'popularity': 0.1}
```

**3. Cascade Approach**

Use approaches in sequence:

```python
def get_recommendations(user_id, num_recs=10):
    # Try collaborative first
    recs = collaborative_filter.recommend(user_id, num_recs)

    # If not enough recommendations (new user/cold-start)
    if len(recs) < num_recs:
        # Fill with content-based
        remaining = num_recs - len(recs)
        user_liked_items = get_user_history(user_id)
        content_recs = content_based_filter.recommend(user_liked_items, remaining)
        recs.extend(content_recs)

    # Still not enough? Fill with popular
    if len(recs) < num_recs:
        remaining = num_recs - len(recs)
        popular_recs = get_popular_items(remaining)
        recs.extend(popular_recs)

    return recs
```

**Interview Tip**: Explain the cascade approach shows you understand graceful degradation.

---

## Diversity and Exploration

### The Problem

Without diversity constraints, recommendations tend to:
- Show only items from user's favorite category
- Recommend very similar items
- Create "filter bubbles"
- Miss cross-selling opportunities

### Solution: Maximal Marginal Relevance (MMR)

MMR balances relevance with diversity:

```python
def mmr_rerank(candidate_items, already_selected, lambda_param=0.5):
    """
    lambda_param: 0 = max diversity, 1 = max relevance
    """
    scores = []
    for item in candidate_items:
        # Relevance score (from recommendation algorithm)
        relevance = item.score

        # Diversity score (dissimilarity to already selected)
        if already_selected:
            max_similarity = max([
                similarity(item, selected)
                for selected in already_selected
            ])
            diversity = 1 - max_similarity
        else:
            diversity = 1.0

        # MMR score
        mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
        scores.append((item, mmr_score))

    # Return item with highest MMR score
    return max(scores, key=lambda x: x[1])
```

**Interview Discussion**: How to choose lambda?
- E-commerce: Usually 0.6-0.8 (favor relevance)
- News/content: 0.4-0.6 (more diversity)
- Discovery platforms: 0.2-0.4 (high diversity)

### Category Diversity

Simple heuristic: Ensure recommendations span multiple categories

```python
def ensure_category_diversity(recommendations, min_categories=3):
    """Ensure recommendations span at least min_categories."""
    selected = []
    categories_seen = set()

    for rec in recommendations:
        category = rec.category
        # Accept if we haven't seen this category OR we have enough diversity
        if category not in categories_seen or len(categories_seen) >= min_categories:
            selected.append(rec)
            categories_seen.add(category)

        if len(selected) >= 10:  # Target number
            break

    return selected
```

---

## Scalability and Performance Optimization

### Challenge

With 10K users and 5K products:
- User-item matrix: 10K × 5K = 50M entries (mostly sparse)
- All-pairs similarity: 5K × 5K = 25M comparisons

### Optimization 1: Pre-computation

**Strategy**: Pre-compute recommendations offline, serve online

```python
# Offline (daily batch job)
for user_id in all_users:
    recommendations = compute_recommendations(user_id)
    cache.set(f"recs:user:{user_id}", recommendations, ttl=86400)  # 24 hours

# Online (< 100ms)
def get_recommendations_fast(user_id):
    recs = cache.get(f"recs:user:{user_id}")
    if recs:
        return recs
    else:
        # Fallback to popular items
        return get_popular_items()
```

**Interview Discussion**: Trade-offs of pre-computation
- **Pros**: Fast serving, predictable latency
- **Cons**: Can't incorporate real-time behavior, storage costs

### Optimization 2: Approximate Nearest Neighbors

For item-item similarity, don't compute all pairs. Use approximate methods:

**Libraries**:
- **FAISS** (Facebook): Highly optimized for similarity search
- **Annoy** (Spotify): Approximate nearest neighbors
- **NMSLIB**: Non-metric space library

**Example with FAISS**:
```python
import faiss

# Build index (offline)
index = faiss.IndexFlatIP(feature_dim)  # Inner product = cosine for normalized vectors
index.add(normalized_item_features)

# Fast similarity search (online)
similarities, indices = index.search(query_item_features, k=10)
```

**Speedup**: O(N) instead of O(N²) for all-pairs

### Optimization 3: Sparse Matrix Operations

Use sparse matrices for user-item interactions:

```python
from scipy.sparse import csr_matrix

# Dense matrix: 10K × 5K × 8 bytes = 400MB
# Sparse matrix (1% density): ~4MB

interactions = csr_matrix((ratings, (user_indices, item_indices)))
```

**Benefits**:
- 100x memory reduction
- Faster matrix operations (skip zeros)
- Essential for larger datasets

### Optimization 4: Caching Strategy

**Multi-level cache**:

```python
# Level 1: In-memory cache (Redis)
# - Popular items (100 products): No TTL
# - Active users (1000 users): 1-hour TTL
# - Regular users: 24-hour TTL

# Level 2: Database cache
# - Pre-computed recommendations for all users
# - Updated daily

# Level 3: Real-time computation
# - Fallback for cache misses
# - New users
```

**Interview Tip**: Show you understand different cache levels and their trade-offs.

---

## Training Strategy

### Data Preparation

**1. Train/Test Split (Time-Based)**

```python
# WRONG: Random split (data leakage!)
train, test = train_test_split(interactions, test_size=0.2)

# CORRECT: Time-based split
cutoff_date = '2024-01-01'
train = interactions[interactions.timestamp < cutoff_date]
test = interactions[interactions.timestamp >= cutoff_date]
```

**Interview Key Point**: Always use time-based splits for temporal data!

**2. Handling Implicit Feedback**

```python
# Convert interactions to implicit feedback
def create_implicit_matrix(interactions):
    # Assign confidence based on interaction type
    confidence_map = {
        'view': 1.0,
        'cart': 3.0,
        'purchase': 5.0
    }

    interactions['confidence'] = interactions['type'].map(confidence_map)

    # Aggregate multiple interactions
    aggregated = interactions.groupby(['user_id', 'product_id']).agg({
        'confidence': 'sum',
        'timestamp': 'max'
    }).reset_index()

    return aggregated
```

### Model Training

**Collaborative Filtering (ALS)**:

```python
from implicit.als import AlternatingLeastSquares

# Create model
model = AlternatingLeastSquares(
    factors=100,        # Number of latent factors
    regularization=0.01, # L2 regularization
    iterations=15,       # Training iterations
    calculate_training_loss=True
)

# Train
model.fit(user_item_matrix)
```

**Hyperparameter Tuning**:
- `factors`: 50-200 (more = better fit, but overfitting risk)
- `regularization`: 0.001-0.1 (prevents overfitting)
- `iterations`: 10-20 (more = better fit, but longer training)

**Interview Discussion**: How to choose hyperparameters?
- Grid search on validation set
- Monitor training loss vs. validation metrics
- Start with defaults, tune if needed

**Content-Based Filtering**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract features
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
text_features = tfidf.fit_transform(products['description'])

# Compute similarity
item_similarity = cosine_similarity(text_features)
```

**No hyperparameters to tune** - deterministic algorithm!

---

## Inference and Serving

### Recommendation Generation

**Input**: User ID, number of recommendations, context

**Output**: Ranked list of product IDs with scores and reasons

**Process**:

```python
def generate_recommendations(user_id, num_recs=10, context='homepage'):
    # 1. Get collaborative filtering recommendations
    collab_recs = collaborative_model.recommend(
        user_id,
        num_recs * 3  # Get more candidates for filtering
    )

    # 2. Get content-based recommendations
    user_liked_items = get_user_history(user_id)
    content_recs = content_based_model.recommend(
        user_liked_items,
        num_recs * 3
    )

    # 3. Get popular items
    popular_recs = get_popular_items(num_recs)

    # 4. Combine with context-aware weights
    weights = get_context_weights(user_id, context)
    hybrid_scores = combine_scores(collab_recs, content_recs, popular_recs, weights)

    # 5. Filter out already purchased items
    hybrid_scores = filter_purchased(user_id, hybrid_scores)

    # 6. Apply diversity (MMR)
    final_recs = mmr_rerank(hybrid_scores, lambda_param=0.7)

    # 7. Add explanations
    final_recs = add_explanations(final_recs)

    return final_recs[:num_recs]
```

### Explainability

Add reasons for each recommendation:

```python
def add_explanations(recommendations, user_id):
    for rec in recommendations:
        # Check which component scored it highest
        if rec.collaborative_score > rec.content_score:
            # Find similar users who liked this
            similar_users = find_similar_users(user_id)
            rec.reason = f"Popular among users like you"

        elif rec.content_score > rec.collaborative_score:
            # Find similar item from user's history
            similar_item = find_most_similar_item(rec.product_id, user_history)
            rec.reason = f"Similar to {similar_item.name}"

        else:
            rec.reason = "Trending in Electronics"

    return recommendations
```

**Interview Key Point**: Explainability builds user trust and helps debugging.

---

## Evaluation Metrics

### Offline Metrics

**1. Precision@K**

Of K recommendations, how many are relevant?

```python
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    num_relevant = len(set(recommended_k) & set(relevant))
    return num_relevant / k
```

**Target**: > 30% for K=10

**2. Recall@K**

Of all relevant items, how many did we recommend?

```python
def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    num_relevant = len(set(recommended_k) & set(relevant))
    return num_relevant / len(relevant) if relevant else 0
```

**Target**: > 20% for K=10

**3. NDCG@K (Normalized Discounted Cumulative Gain)**

Accounts for ranking - higher-ranked relevant items score better.

```python
def ndcg_at_k(recommended, relevant, k):
    dcg = sum([
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    ])

    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])

    return dcg / idcg if idcg > 0 else 0
```

**Target**: > 0.4 for K=10

**4. Coverage**

What % of products get recommended?

```python
def catalog_coverage(all_recommendations, catalog_size):
    unique_recommended = set(all_recommendations)
    return len(unique_recommended) / catalog_size
```

**Target**: > 50% (avoid only recommending popular items)

**5. Diversity**

Average dissimilarity between recommended items:

```python
def diversity(recommendations):
    total_distance = 0
    count = 0

    for i, item1 in enumerate(recommendations):
        for item2 in recommendations[i+1:]:
            total_distance += 1 - similarity(item1, item2)
            count += 1

    return total_distance / count if count > 0 else 0
```

### Online Metrics (A/B Testing)

**1. Click-Through Rate (CTR)**

```python
CTR = (clicks on recommendations) / (total recommendations shown)
```

**Target**: > 5%

**2. Conversion Rate**

```python
Conversion = (purchases from recommendations) / (clicks on recommendations)
```

**Target**: > 2%

**3. Average Order Value (AOV)**

```python
AOV = total_revenue / number_of_orders
```

**Target**: 10% increase

**Interview Discussion**: Why online metrics matter more than offline?
- Offline metrics don't capture user satisfaction
- Selection bias in historical data
- Real business impact vs. academic metrics

---

## Deployment Architecture

### System Components

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────────┐
│  API Gateway    │
│  (Load Balancer)│
└──────┬──────────┘
       │
       ▼
┌──────────────────────┐
│ Recommendation API   │
│   (FastAPI/Flask)    │
└──────┬───────────────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌──────────┐   ┌─────────────┐
│  Cache   │   │ Rec Engine  │
│ (Redis)  │   │ (ML Models) │
└──────────┘   └──────┬──────┘
                      │
                      ▼
               ┌──────────────┐
               │  Database    │
               │ (PostgreSQL) │
               └──────────────┘
```

### API Design

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    context: str = "homepage"
    exclude_purchased: bool = True

class RecommendationResponse(BaseModel):
    product_id: str
    score: float
    reason: str

@app.post("/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    try:
        # Check cache first
        cache_key = f"recs:{request.user_id}:{request.context}"
        cached = await cache.get(cache_key)

        if cached:
            return cached

        # Generate recommendations
        recs = recommendation_engine.generate(
            user_id=request.user_id,
            num_recs=request.num_recommendations,
            context=request.context
        )

        # Cache results
        await cache.set(cache_key, recs, ttl=3600)

        return recs

    except Exception as e:
        # Log error
        logger.error(f"Error generating recommendations: {e}")

        # Fallback to popular items
        return get_popular_items(request.num_recommendations)
```

**Interview Key Points**:
- Always have fallback (popular items)
- Cache for performance
- Graceful error handling
- Return response even if partial failure

---

## Monitoring and Maintenance

### Metrics to Monitor

**1. System Metrics**
- Latency (p50, p95, p99)
- Error rate
- Cache hit rate
- API throughput

**2. Model Metrics**
- CTR (daily)
- Conversion rate (daily)
- Average order value (daily)
- Coverage (weekly)
- Diversity (weekly)

**3. Data Quality Metrics**
- Interaction volume (daily active users)
- New users/products (daily)
- Sparsity of interaction matrix

### Model Updates

**When to retrain**:
- Scheduled (daily/weekly batch)
- Performance degradation (CTR drops > 10%)
- New data patterns (seasonal changes)

**How to update**:
```python
# Daily incremental update
new_interactions = get_interactions_since(last_update)
model.partial_fit(new_interactions)  # Incremental learning

# Weekly full retrain
model = train_from_scratch(all_interactions)
```

### A/B Testing

**Scenario**: Test new algorithm vs. current

```python
def assign_user_to_treatment(user_id):
    # Consistent hashing for stable assignment
    hash_value = hash(user_id) % 100

    if hash_value < 50:
        return "control"  # Current algorithm
    else:
        return "treatment"  # New algorithm

# Serve recommendations based on treatment
treatment = assign_user_to_treatment(user_id)
if treatment == "control":
    recs = current_algorithm.recommend(user_id)
else:
    recs = new_algorithm.recommend(user_id)
```

**Evaluation**:
- Run for 2-4 weeks
- Monitor CTR, conversion, revenue
- Check for statistical significance (p < 0.05)
- Consider business metrics, not just model metrics

---

## Trade-offs and Design Decisions

### 1. Collaborative vs. Content-Based

| Decision | Rationale |
|----------|-----------|
| Use hybrid | Handles cold-start for both users and items |
| Favor collaborative for active users | More personalized, discovers unexpected patterns |
| Favor content for new items | Can recommend immediately with metadata |

### 2. Matrix Factorization vs. Deep Learning

| Approach | Pros | Cons | Our Choice |
|----------|------|------|------------|
| Matrix Factorization (ALS) | Fast, interpretable, works well | Limited feature integration | **Chosen** - Good baseline |
| Deep Learning (NCF) | Can learn complex patterns | Needs more data, slower | Future improvement |

**Interview Tip**: Start simple, mention deep learning as extension.

### 3. Pre-computation vs. Real-time

| Approach | Latency | Freshness | Our Choice |
|----------|---------|-----------|------------|
| Pre-compute all | Very low (<10ms) | Stale (24hrs) | **Homepage** |
| Compute on-demand | High (>500ms) | Fresh | Expensive |
| Hybrid (pre-compute + adjust) | Low (<100ms) | Semi-fresh | **Best of both** |

**Our Strategy**:
- Pre-compute base recommendations daily
- Adjust scores based on recent activity (last hour)
- Cache for 1 hour

### 4. Accuracy vs. Diversity

| Approach | Accuracy | Diversity | User Experience |
|----------|----------|-----------|-----------------|
| Pure relevance | Highest | Lowest | Filter bubble |
| Pure diversity | Lowest | Highest | Irrelevant items |
| MMR (λ=0.7) | High | Medium | **Best balance** |

### 5. Explicit vs. Implicit Feedback

**Our Choice**: Implicit feedback (views, purchases)

**Rationale**:
- More data available (users don't rate products)
- Reflects actual behavior (not stated preference)
- Can handle with confidence weighting

**Trade-off**: Noisier signal (view ≠ like), but more data compensates.

---

## Common Interview Follow-up Questions

### Q1: How would you handle seasonality?

**Answer**:
- Time-decay weighting for interactions (recent = more important)
- Separate models for different seasons
- Trending items component with short time windows
- Boost seasonal categories (e.g., "gifts" in December)

### Q2: How would you scale to 10 million users?

**Answer**:
- Distributed computation (Spark) for model training
- Approximate nearest neighbors (FAISS) for similarity search
- Sharded caching (partition users across cache servers)
- Asynchronous batch updates
- Pre-compute only for active users, popular items for others

### Q3: How would you handle privacy concerns?

**Answer**:
- Don't store unnecessary personal data
- Aggregate interaction data
- Allow users to delete their data (GDPR)
- Differential privacy for collaborative filtering
- Federated learning (future)

### Q4: How would you explain recommendations to users?

**Answer**:
- Track which component scored each recommendation highest
- "Because you purchased X" (collaborative)
- "Similar to X" (content-based)
- "Trending in category Y" (popularity)
- Allow users to say "not interested" for feedback

### Q5: How would you handle cold-start for both users AND items?

**Answer**:
- **Cold user + cold item**: Impossible to personalize
  - Show popular items in same category
  - Or ask user for preferences
- **Cold user + existing item**: Content-based or popular
- **Existing user + cold item**: Content-based recommendations to relevant users

---

## Implementation Summary

### Core Components (Interview Scope)

1. **Data Generation** (~100 lines)
   - Synthetic users, products, interactions
   - Export to CSV/JSON

2. **Collaborative Filter** (~150 lines)
   - ALS matrix factorization
   - User-item interaction matrix
   - Recommendation generation

3. **Content-Based Recommender** (~150 lines)
   - TF-IDF feature extraction
   - Cosine similarity
   - Similar item recommendations

4. **Hybrid System** (~100 lines)
   - Weighted combination
   - Context-aware switching
   - Diversity injection (MMR)

5. **Evaluation** (~100 lines)
   - Precision@K, Recall@K, NDCG@K
   - Coverage, Diversity
   - Visualization

**Total**: ~600 lines of clean, commented Python code demonstrating understanding.

---

## Key Takeaways for Interviews

1. **Start with clarifying questions**: Data availability, scale, latency requirements

2. **Discuss multiple approaches**: Collaborative, content-based, hybrid - explain trade-offs

3. **Address cold-start explicitly**: This is the #1 challenge, have concrete solutions

4. **Think about scale**: Pre-computation, caching, approximate methods

5. **Evaluation matters**: Offline metrics (precision@K) AND online metrics (CTR, revenue)

6. **Be practical**: Balance accuracy with latency, explainability, and business constraints

7. **Show system thinking**: Not just algorithms, but deployment, monitoring, A/B testing

**Remember**: There's no single "right" answer. Show you can reason about trade-offs and make informed decisions based on requirements.
