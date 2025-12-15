# E-commerce Recommendations - Problem Statement

## Overview

Design and implement a recommendation system for an e-commerce platform that suggests relevant products to users based on their browsing history, purchase patterns, and product characteristics. This is one of the most common ML system design interview questions, as recommendation systems are fundamental to many tech companies (Amazon, Netflix, Spotify, etc.).

**Interview Context**: This problem tests your understanding of different recommendation approaches, their trade-offs, and how to handle real-world challenges like cold-start problems, scalability, and evaluation metrics.

---

## Business Context

You're building a recommendation system for **ShopSmart**, a mid-sized e-commerce platform with:
- **10,000+ active users** per day
- **5,000+ products** across 20+ categories
- **User interactions**: product views, add-to-cart, purchases, ratings
- **Goal**: Increase user engagement, average order value, and conversion rate

The business wants to show personalized product recommendations in three places:
1. **Homepage**: "Recommended for You" section (10 products)
2. **Product Page**: "You May Also Like" (5 products)
3. **Cart**: "Frequently Bought Together" (3 products)

---

## Problem Requirements

### Functional Requirements

1. **Personalized Recommendations**
   - Generate product recommendations tailored to each user
   - Consider user's browsing history, purchases, and ratings
   - Recommend products the user hasn't already purchased

2. **Cold-Start Handling**
   - **New users**: Recommend popular items or based on demographic info
   - **New products**: Recommend based on product features/category
   - This is a critical interview discussion point

3. **Diversity and Serendipity**
   - Don't just recommend similar products (avoid filter bubbles)
   - Include some exploratory recommendations
   - Balance relevance with discovery

4. **Multiple Contexts**
   - Homepage recommendations (personalized)
   - Similar product recommendations (content-based)
   - Complementary products (association rules)

### Non-Functional Requirements

1. **Latency**: < 100ms for recommendation retrieval
2. **Scalability**: Handle 10K+ users with growing catalog
3. **Explainability**: Ability to explain why items were recommended
4. **Real-time Updates**: Incorporate recent user behavior

---

## Technical Constraints

1. **Available Data**
   - User-product interactions (implicit and explicit feedback)
   - Product metadata (category, brand, price, description)
   - User demographics (optional)
   - Timestamps for all interactions

2. **System Constraints**
   - Recommendations must be pre-computed or very fast
   - Storage for user-item matrices and model artifacts
   - Ability to update recommendations periodically

3. **Interview Scope**
   - Focus on core recommendation algorithms
   - Discuss deployment strategy, not implement full production system
   - Emphasize understanding of trade-offs

---

## Input/Output Specification

### Input

**User Interactions**:
```python
{
    "user_id": "U12345",
    "product_id": "P98765",
    "interaction_type": "purchase",  # view, cart, purchase, rating
    "rating": 5,  # optional, 1-5 scale
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**Product Metadata**:
```python
{
    "product_id": "P98765",
    "name": "Wireless Bluetooth Headphones",
    "category": "Electronics > Audio",
    "brand": "SoundTech",
    "price": 79.99,
    "description": "High-quality wireless headphones with noise cancellation...",
    "tags": ["wireless", "bluetooth", "noise-canceling"]
}
```

### Output

**Recommendation Request**:
```python
{
    "user_id": "U12345",
    "num_recommendations": 10,
    "context": "homepage",  # homepage, product_page, cart
    "exclude_purchased": True
}
```

**Recommendation Response**:
```python
{
    "user_id": "U12345",
    "recommendations": [
        {
            "product_id": "P11111",
            "score": 0.92,
            "reason": "Based on your purchase of Wireless Headphones"
        },
        {
            "product_id": "P22222",
            "score": 0.87,
            "reason": "Popular in Electronics category"
        },
        ...
    ]
}
```

---

## Key Interview Challenges

### 1. Cold-Start Problem

**Challenge**: How do you recommend products to new users or recommend new products?

**Expected Discussion**:
- New users: Use popularity-based, demographic-based, or ask for preferences
- New products: Use content-based features, promote to early adopters
- Hybrid approaches that gracefully degrade

**Interview Tip**: This is the #1 question asked about recommendation systems. Be ready with multiple solutions.

### 2. Scalability

**Challenge**: How do you handle millions of users and products?

**Expected Discussion**:
- Matrix factorization vs. neural approaches
- Pre-computation and caching strategies
- Approximate nearest neighbor search
- Distributed computation

### 3. Implicit vs. Explicit Feedback

**Challenge**: Most e-commerce data is implicit (views, clicks) not explicit (ratings).

**Expected Discussion**:
- How to weight different interaction types
- Dealing with sparse data
- Negative sampling strategies

### 4. Evaluation

**Challenge**: How do you measure if recommendations are good?

**Expected Discussion**:
- Offline metrics: precision@k, recall@k, NDCG
- Online metrics: CTR, conversion rate, revenue
- A/B testing considerations

---

## Success Metrics

### Model Performance Metrics

1. **Precision@K**: Of K recommendations, how many are relevant?
   - Target: > 30% for K=10

2. **Recall@K**: Of all relevant items, how many did we recommend?
   - Target: > 20% for K=10

3. **NDCG@K**: Normalized Discounted Cumulative Gain (considers ranking)
   - Target: > 0.4 for K=10

4. **Coverage**: % of product catalog that gets recommended
   - Target: > 50% (avoid recommending only popular items)

5. **Diversity**: Average dissimilarity between recommended items
   - Measure using category diversity or feature distance

### Business Metrics

1. **Click-Through Rate (CTR)**: % of recommendations clicked
   - Target: > 5%

2. **Conversion Rate**: % of recommendation clicks leading to purchase
   - Target: > 2%

3. **Average Order Value (AOV)**: Revenue per transaction
   - Target: 10% increase

4. **User Engagement**: Time on site, pages per session
   - Target: 15% increase

---

## Evaluation Scenarios

### Scenario 1: Active User with Purchase History

**Input**:
- User has purchased 10 items in "Electronics" and "Books"
- Recent views: smartphone accessories
- Never purchased in "Home & Garden"

**Expected Behavior**:
- Recommend electronics accessories and related books
- Maybe one exploratory item from a new category
- High confidence predictions

### Scenario 2: New User (Cold-Start)

**Input**:
- User just signed up, no interaction history
- Demographic info: age 25-34, urban location

**Expected Behavior**:
- Show trending/popular items
- If demographic info available, show category preferences for similar users
- Ask for explicit preferences (optional)

### Scenario 3: Product Page Context

**Input**:
- User viewing "iPhone 14 Pro"
- Context: product detail page

**Expected Behavior**:
- Show complementary items: cases, screen protectors, chargers
- Show similar smartphones (if content-based)
- Show "frequently bought together" items

---

## Common Interview Questions

### Algorithmic Questions

1. **Explain collaborative filtering vs. content-based filtering**
   - When would you use each?
   - What are the trade-offs?

2. **How do you handle the cold-start problem?**
   - For new users?
   - For new items?

3. **What is matrix factorization? How does it work?**
   - SVD vs. ALS vs. neural approaches
   - How do you choose the number of latent factors?

4. **How do you incorporate implicit feedback?**
   - Weighting different interaction types
   - Negative sampling

5. **How would you build a hybrid recommender?**
   - Weighted combination
   - Feature augmentation
   - Ensemble methods

### System Design Questions

1. **How do you scale recommendations to millions of users?**
   - Pre-computation strategies
   - Caching layers
   - Approximate nearest neighbors

2. **How do you serve recommendations with < 100ms latency?**
   - Model serving architecture
   - Indexing strategies

3. **How do you update recommendations in real-time?**
   - Online learning vs. batch updates
   - Feature computation pipeline

4. **How do you handle seasonality and trends?**
   - Time-aware features
   - Decay factors for older interactions

### Evaluation Questions

1. **How do you evaluate recommendations offline?**
   - Train/test splits for temporal data
   - Metrics: precision@k, recall@k, NDCG

2. **What are the limitations of offline evaluation?**
   - Doesn't capture user satisfaction
   - Selection bias in historical data

3. **How do you run A/B tests for recommendations?**
   - Control vs. treatment groups
   - Key metrics to track
   - Statistical significance

### Business Questions

1. **How do you balance accuracy with diversity?**
   - MMR (Maximal Marginal Relevance)
   - Post-processing for diversity

2. **How do you explain recommendations to users?**
   - "Because you bought X"
   - Transparency and trust

3. **How do you handle business constraints?**
   - Promoting certain products
   - Filtering out-of-stock items
   - Regional availability

---

## Extensions and Follow-ups

If you finish the basic implementation, interviewers might ask about:

1. **Multi-armed Bandits**
   - Exploration vs. exploitation
   - Thompson sampling, UCB

2. **Session-Based Recommendations**
   - RNNs for sequential patterns
   - Session context

3. **Context-Aware Recommendations**
   - Time of day, device, location
   - Contextual bandits

4. **Deep Learning Approaches**
   - Neural collaborative filtering
   - Two-tower models
   - Graph neural networks

5. **Fairness and Bias**
   - Popularity bias
   - Filter bubbles
   - Fair exposure for sellers

---

## Resources

### Algorithms and Approaches

1. **Collaborative Filtering**
   - User-based CF
   - Item-based CF
   - Matrix factorization (SVD, ALS, NMF)

2. **Content-Based Filtering**
   - TF-IDF for text features
   - Cosine similarity
   - Feature engineering

3. **Hybrid Methods**
   - Weighted combination
   - Feature augmentation
   - Cascade approaches

### Key Papers

1. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering"
2. "Matrix Factorization Techniques for Recommender Systems"
3. "Neural Collaborative Filtering"

### Libraries and Tools

- **Surprise**: Python library for collaborative filtering
- **LightFM**: Hybrid recommendation algorithms
- **scikit-learn**: For content-based approaches
- **Implicit**: For implicit feedback datasets
- **FAISS**: For fast similarity search

---

## Time Expectations

### 45-Minute Interview
- 5 min: Clarify requirements
- 10 min: Discuss approach and trade-offs
- 20 min: Design system architecture
- 10 min: Discuss evaluation and improvements

### 2-Hour Coding Interview
- 15 min: Requirements and approach
- 60 min: Implement basic recommender
- 30 min: Discuss evaluation and extensions
- 15 min: Code review and improvements

### Take-Home Assignment (4-6 hours)
- Implement collaborative + content-based recommenders
- Evaluate on offline metrics
- Write up approach and trade-offs
- Create visualizations

---

## Interview Tips

1. **Start with Clarifying Questions**
   - What data is available?
   - What's the scale (users, products, interactions)?
   - What are the latency requirements?
   - Is explainability important?

2. **Discuss Multiple Approaches**
   - Don't jump to one solution
   - Compare collaborative filtering, content-based, and hybrid
   - Explain trade-offs

3. **Address Cold-Start Explicitly**
   - This is the most important challenge
   - Have 2-3 concrete solutions ready

4. **Think About Scale**
   - Pre-computation strategies
   - Caching layers
   - Approximate methods

5. **Evaluation is Key**
   - Discuss offline and online metrics
   - Mention A/B testing
   - Talk about business metrics, not just model metrics

6. **Be Practical**
   - Balance accuracy with latency
   - Consider data availability
   - Think about maintenance and monitoring

---

## Summary

This problem tests your understanding of:
- Different recommendation algorithms and when to use them
- Cold-start problem solutions
- Scalability and performance considerations
- Evaluation metrics and A/B testing
- Business context and trade-offs

**Key Takeaway**: There's no single "right" approach. The best recommendation system depends on the data, scale, latency requirements, and business objectives. Interviewers want to see that you can reason about these trade-offs and make informed decisions.
