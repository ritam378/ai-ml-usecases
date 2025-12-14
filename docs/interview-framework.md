# ML Interview Framework

A systematic approach to tackling machine learning case study interviews at FAANG and top tech companies.

## Table of Contents
1. [Interview Types](#interview-types)
2. [General Framework](#general-framework)
3. [Problem Clarification](#problem-clarification)
4. [Solution Design](#solution-design)
5. [Implementation Discussion](#implementation-discussion)
6. [Production Considerations](#production-considerations)
7. [Common Pitfalls](#common-pitfalls)

## Interview Types

### 1. ML Case Study (60-90 minutes)
- Design an ML system for a business problem
- Focus on problem formulation, metrics, and high-level approach
- Less coding, more system design

### 2. Coding + ML (45-60 minutes)
- Implement a specific ML algorithm or component
- Write production-quality code
- Explain trade-offs

### 3. ML System Design (45-60 minutes)
- End-to-end ML pipeline design
- Scalability, latency, cost considerations
- Similar to distributed systems design

## General Framework

### Step 1: Clarify the Problem (5-10 minutes)

**Ask about:**
- Business objective: What are we trying to achieve?
- Success metrics: How do we measure success?
- Constraints: Latency, cost, interpretability requirements?
- Data availability: What data do we have access to?
- Scale: Number of users, requests per second, data volume?
- User experience: Real-time vs. batch predictions?

**Example Questions:**
```
"What's the primary business goal? Are we optimizing for revenue, engagement, or something else?"
"What's the acceptable latency for predictions?"
"Do we need the model to be interpretable for regulatory reasons?"
"What's our current data infrastructure?"
```

### Step 2: Frame as ML Problem (5-10 minutes)

**Define:**
- Problem type: Classification, regression, ranking, clustering, etc.
- Input features: What signals can we use?
- Output: What are we predicting?
- Success metrics: Accuracy, precision, recall, F1, AUC, NDCG, etc.

**Example:**
```
Business: Reduce credit card fraud
ML Problem: Binary classification (fraud vs. legitimate)
Input: Transaction features (amount, merchant, location, time, user history)
Output: Probability of fraud (0-1)
Metrics: Precision/Recall (optimize for recall to catch fraud, balance with precision to avoid false positives)
```

### Step 3: Discuss Data (5-10 minutes)

**Cover:**
- Data sources: Where does the data come from?
- Data quality: Missing values, noise, outliers?
- Labeling: How are labels obtained? Label quality?
- Class imbalance: Are classes balanced?
- Data drift: Does data distribution change over time?
- Privacy/compliance: Any PII or sensitive data concerns?

**Key Points:**
```
"For fraud detection, we likely have severe class imbalance (0.1% fraud rate).
We'll need to address this through:
- Oversampling/undersampling
- Cost-sensitive learning
- Anomaly detection approaches
```

### Step 4: Feature Engineering (5-10 minutes)

**Discuss:**
- Raw features: Transaction amount, merchant category, location
- Derived features:
  - User aggregates: Avg transaction amount, transaction frequency
  - Time-based: Hour of day, day of week, time since last transaction
  - Ratios: Current amount / avg amount
  - Sequences: Recent transaction history
- Feature transformations: Normalization, encoding, binning

**Example:**
```
Key Features for Fraud Detection:
1. Transaction amount (normalized by user's average)
2. Merchant category
3. Distance from user's typical locations
4. Time since last transaction
5. Number of transactions in last hour
6. Velocity features (rapid succession of transactions)
```

### Step 5: Model Selection (5-10 minutes)

**Consider:**
- Simple baseline: Logistic regression, decision tree
- Advanced models: Random Forest, XGBoost, neural networks
- Trade-offs: Accuracy vs. interpretability vs. latency

**Framework for Discussion:**
```
1. Start with baseline: "I'd start with logistic regression as a baseline"
2. Discuss advanced options: "Then move to gradient boosting (XGBoost) for better performance"
3. Explain trade-offs: "Deep learning could work but adds complexity and latency"
4. Choose based on constraints: "Given the interpretability requirement, I'd recommend XGBoost with SHAP values"
```

### Step 6: Training Strategy (5-10 minutes)

**Cover:**
- Train/validation/test split: Time-based split for temporal data
- Cross-validation: K-fold or time-series CV
- Hyperparameter tuning: Grid search, random search, Bayesian optimization
- Handling class imbalance: SMOTE, class weights, threshold tuning
- Regularization: L1/L2, dropout, early stopping

**Example:**
```
Training Strategy for Fraud Detection:
1. Time-based split: Train on Jan-Mar, validate on April, test on May
2. Handle imbalance: Use class_weight='balanced' in XGBoost
3. Tune threshold: Optimize for desired precision/recall balance
4. Cross-validation: 5-fold CV on training set
```

### Step 7: Evaluation (5-10 minutes)

**Discuss:**
- Offline metrics: Precision, recall, F1, AUC-ROC, AUC-PR
- Online metrics: A/B test metrics (fraud caught, false positive rate)
- Business metrics: Revenue saved, customer satisfaction
- Error analysis: Analyze false positives and false negatives

**Evaluation Framework:**
```
Offline Evaluation:
- AUC-PR (better for imbalanced data than AUC-ROC)
- Precision-Recall curve
- Confusion matrix at different thresholds

Online Evaluation:
- A/B test: Treatment (new model) vs. Control (current system)
- Monitor: Fraud caught, false positive rate, customer complaints
- Ramp up gradually: 1% -> 5% -> 25% -> 100%
```

### Step 8: Deployment & Monitoring (5-10 minutes)

**Cover:**
- Serving infrastructure: Online (API) vs. batch predictions
- Latency requirements: Real-time (<100ms) vs. near-real-time (seconds)
- Scalability: Requests per second, peak traffic handling
- Monitoring: Model performance, data drift, feature distribution
- Retraining: Frequency, triggers, automation

**Production Checklist:**
```
1. Deployment:
   - Model serving: REST API with FastAPI or gRPC
   - Load balancing and auto-scaling
   - Feature store for consistent features

2. Monitoring:
   - Prediction latency (p50, p95, p99)
   - Model metrics (precision, recall) over time
   - Feature distribution shifts
   - Upstream data quality

3. Retraining:
   - Schedule: Weekly or when performance degrades
   - Automated pipeline: Data -> Train -> Evaluate -> Deploy
   - A/B test before full rollout
```

## Problem Clarification

### Key Questions to Ask

#### Business Context
- What problem are we solving?
- Who are the stakeholders?
- What's the current solution?
- What's the expected impact?

#### Success Metrics
- How do we define success?
- What's the primary metric?
- Any secondary metrics?
- Trade-offs between metrics?

#### Constraints
- Latency requirements?
- Cost constraints?
- Interpretability needs?
- Regulatory requirements?

#### Data
- What data is available?
- How is it collected?
- Data volume and velocity?
- Labeling process?

#### Scale
- Number of users?
- Requests per second?
- Data size?
- Growth expectations?

## Solution Design

### Template for Discussion

```
1. Problem Framing
   "This is a [classification/regression/ranking] problem where we predict [output] from [input]"

2. High-Level Approach
   "I'd approach this with a [pipeline description]"

3. Data Pipeline
   "We'll need to [collect/process/feature engineer]"

4. Model Selection
   "Starting with [baseline], scaling to [advanced model] because [reasoning]"

5. Evaluation Strategy
   "Measure success using [metrics] with [validation approach]"

6. Production Considerations
   "Deploy as [serving pattern] with [monitoring] and [retraining strategy]"
```

## Implementation Discussion

### Code Structure

When asked to implement:

```python
# 1. Clear function signature with types
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'xgboost'
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a classification model with cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train

    Returns:
        Trained model and validation metrics
    """
    # Implementation
    pass

# 2. Discuss trade-offs
# "I'm using XGBoost because it handles non-linear relationships well
#  and provides feature importance. Trade-off is longer training time
#  compared to logistic regression."

# 3. Handle edge cases
# "We should handle missing values, check for data leakage,
#  and validate input types."
```

### Key Implementation Areas

1. **Data Preprocessing**
   - Missing value handling
   - Outlier detection
   - Feature scaling
   - Encoding categorical variables

2. **Feature Engineering**
   - Creating derived features
   - Feature selection
   - Dimensionality reduction

3. **Model Training**
   - Train/validation split
   - Cross-validation
   - Hyperparameter tuning
   - Handling imbalance

4. **Evaluation**
   - Metrics calculation
   - Error analysis
   - Visualization

## Production Considerations

### Deployment Patterns

#### 1. Batch Prediction
```
Use Cases: Non-time-sensitive predictions
Examples: Daily product recommendations, weekly churn predictions

Pros:
+ Simple infrastructure
+ Easy to debug
+ Cost-effective

Cons:
- Not real-time
- Stale predictions
```

#### 2. Online Prediction (API)
```
Use Cases: Real-time decisions
Examples: Fraud detection, ad ranking

Pros:
+ Real-time predictions
+ Fresh model outputs
+ Better user experience

Cons:
- More complex infrastructure
- Latency requirements
- Higher cost
```

#### 3. Hybrid Approach
```
Use Cases: Balance between real-time and batch
Examples: Pre-compute candidate sets (batch), re-rank in real-time

Pros:
+ Combines benefits of both
+ Flexible

Cons:
- More complex
- Requires coordination
```

### Monitoring Strategy

#### Model Performance
- Track key metrics (precision, recall, AUC) over time
- Set up alerts for degradation
- Compare to baseline

#### Data Quality
- Monitor feature distributions
- Detect anomalies in input data
- Track missing value rates

#### System Health
- Prediction latency (p50, p95, p99)
- Throughput (QPS)
- Error rates
- Resource utilization

#### Business Metrics
- Revenue impact
- User engagement
- Customer satisfaction

### Retraining Strategy

```
Triggers for Retraining:
1. Scheduled: Weekly, monthly
2. Performance-based: When metrics drop below threshold
3. Data-based: When significant distribution shift detected

Retraining Process:
1. Collect new labeled data
2. Retrain model with new data
3. Evaluate on holdout set
4. A/B test new model vs. current
5. Gradual rollout if successful
6. Monitor and iterate
```

## Common Pitfalls

### 1. Jumping to Solution Too Quickly
**Problem:** Starting to code or design without understanding requirements
**Solution:** Spend time clarifying the problem, metrics, and constraints

### 2. Ignoring Business Context
**Problem:** Focusing only on ML metrics, ignoring business impact
**Solution:** Connect ML metrics to business outcomes

### 3. Over-engineering
**Problem:** Proposing complex deep learning when simple model would work
**Solution:** Start simple, justify added complexity

### 4. Ignoring Data Issues
**Problem:** Assuming clean, balanced, well-labeled data
**Solution:** Discuss data quality, labeling, imbalance

### 5. Not Considering Production
**Problem:** Designing solution that can't be deployed
**Solution:** Discuss latency, scalability, monitoring from the start

### 6. Poor Communication
**Problem:** Not explaining thought process clearly
**Solution:** Think out loud, structure your approach

### 7. Ignoring Trade-offs
**Problem:** Not discussing pros/cons of choices
**Solution:** For every decision, mention alternatives and trade-offs

### 8. Not Asking Questions
**Problem:** Making assumptions without clarification
**Solution:** Ask clarifying questions throughout

## Interview Tips

### Do's
- Think out loud - explain your reasoning
- Ask clarifying questions
- Structure your approach (use the framework)
- Discuss trade-offs for every decision
- Consider the business context
- Be honest about what you don't know
- Listen to interviewer hints

### Don'ts
- Don't jump to coding immediately
- Don't assume perfect data
- Don't ignore constraints (latency, cost)
- Don't propose solutions you can't explain
- Don't be afraid to change your approach
- Don't forget to summarize at the end

### Time Management

**For 60-minute interview:**
- Problem clarification: 10 minutes
- Solution design: 20 minutes
- Implementation details: 20 minutes
- Questions and discussion: 10 minutes

**For 90-minute interview:**
- Problem clarification: 15 minutes
- Solution design: 30 minutes
- Implementation details: 30 minutes
- Questions and discussion: 15 minutes

## Example Walkthrough

### Problem: Design a recommendation system for e-commerce

**Step 1: Clarify (10 min)**
```
Q: What products are we recommending? How many SKUs?
A: General e-commerce, 100K products

Q: Where do recommendations appear? Home page, product page, email?
A: Product page - "You may also like"

Q: What's the latency requirement?
A: <100ms for page load

Q: How many recommendations to show?
A: Top 10 products

Q: What data do we have?
A: User behavior (clicks, purchases), product metadata, user demographics
```

**Step 2: Frame as ML Problem (5 min)**
```
Problem Type: Ranking/Recommendation
Input: User context + current product + historical behavior
Output: Ranked list of product IDs
Metrics:
- Offline: Recall@10, NDCG@10
- Online: CTR, conversion rate, revenue per user
```

**Step 3: Data Discussion (5 min)**
```
Data Sources:
- User-product interactions (clicks, purchases)
- Product catalog (category, price, description)
- User profile (age, location, past purchases)

Challenges:
- Cold start: New users, new products
- Sparsity: Most user-product pairs have no interaction
- Temporal: User preferences change over time
```

**Step 4: Feature Engineering (5 min)**
```
User Features:
- Demographics: age, location
- Aggregates: avg order value, purchase frequency
- Sequences: last 10 products viewed

Product Features:
- Metadata: category, price, brand
- Popularity: view count, purchase count
- Temporal: trending score

User-Product Interactions:
- Same category as current product
- Price ratio
- Collaborative signals
```

**Step 5: Model Selection (10 min)**
```
Approach: Two-stage ranking

Stage 1: Candidate Generation (Recall)
- Collaborative filtering (matrix factorization)
- Content-based (same category, brand)
- Generate 100-500 candidates

Stage 2: Ranking (Precision)
- Gradient boosting (XGBoost)
- Deep learning (two-tower model)
- Rank top 100 candidates to get top 10

Justification:
- Two-stage allows low latency (<100ms)
- Collaborative filtering handles cold start with content-based backup
- XGBoost provides good precision with interpretability
```

**Step 6: Training (5 min)**
```
Training Data:
- Positive: User purchased product within 7 days of viewing
- Negative: Sampled from non-purchases

Training Strategy:
- Time-based split (train on last 3 months, validate on next 2 weeks)
- Handle imbalance: Negative sampling ratio 10:1
- Hyperparameter tuning with Bayesian optimization
```

**Step 7: Evaluation (10 min)**
```
Offline:
- Recall@100 for candidate generation
- NDCG@10 for final ranking
- Analyze by category, price range, user segment

Online:
- A/B test: New model vs. current system
- Metrics: CTR (+5% target), conversion rate, revenue
- Segment analysis: New vs. returning users
```

**Step 8: Production (10 min)**
```
Deployment:
- Candidate generation: Precompute daily, store in cache
- Ranking: Real-time via REST API
- Latency: 50ms p95 target

Monitoring:
- Model: NDCG, coverage (% products recommended)
- System: Latency, throughput, error rate
- Business: CTR, conversion, revenue

Retraining:
- Weekly: Incorporate new interaction data
- Trigger: If CTR drops >10% or NDCG drops >5%
```

## Summary

The key to ML interview success:
1. **Structured approach**: Follow the framework
2. **Communication**: Think out loud, explain reasoning
3. **Breadth and depth**: Cover all aspects but go deep on key areas
4. **Practical focus**: Consider production realities
5. **Trade-offs**: Discuss pros/cons of every choice

Good luck with your interviews!
