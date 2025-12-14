# ML System Design Guide

A comprehensive guide to designing production machine learning systems for senior-level interviews.

## Table of Contents
1. [Overview](#overview)
2. [System Design Framework](#system-design-framework)
3. [Common ML System Patterns](#common-ml-system-patterns)
4. [Component Deep Dives](#component-deep-dives)
5. [Case Studies](#case-studies)
6. [Interview Tips](#interview-tips)

## Overview

ML system design interviews evaluate your ability to:
- Design end-to-end ML systems
- Make trade-offs between conflicting requirements
- Scale systems to production needs
- Consider operational aspects (monitoring, retraining)

**Key Differences from Software System Design:**
- Data is a first-class citizen
- Model training and serving are separate concerns
- Performance degrades over time (drift)
- Experimentation and iteration are critical

## System Design Framework

### Step 1: Requirements Gathering (10-15 min)

#### Functional Requirements
- What problem are we solving?
- What are the inputs and outputs?
- What's the user experience?

#### Non-Functional Requirements
- **Scale**: Users, requests/sec, data volume
- **Latency**: Real-time (<100ms), near-real-time (<1s), batch
- **Availability**: 99.9%, 99.99%, 99.999%
- **Cost**: Budget constraints, cost per prediction
- **Accuracy**: Minimum acceptable performance
- **Interpretability**: Explainability requirements

#### Constraints
- Existing infrastructure
- Team expertise
- Regulatory requirements
- Privacy concerns

### Step 2: Problem Formulation (5-10 min)

- Frame as ML problem (classification, ranking, etc.)
- Define success metrics
- Identify data sources
- Discuss labeling strategy

### Step 3: High-Level Architecture (10-15 min)

Draw and discuss:
```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│     API Gateway/LB          │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐      ┌────────────────┐
│  Prediction Service         │──────│  Feature Store │
│  (Model Serving)            │      └────────────────┘
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Monitoring & Logging       │
└─────────────────────────────┘

┌─────────────────────────────┐
│  Training Pipeline          │
│  (Offline)                  │
│  ┌────────────────────┐     │      ┌────────────────┐
│  │ Data Ingestion     │─────┼─────│  Data Storage  │
│  └────────┬───────────┘     │      └────────────────┘
│           │                 │
│           ▼                 │
│  ┌────────────────────┐     │
│  │ Feature Engineering │     │
│  └────────┬───────────┘     │
│           │                 │
│           ▼                 │
│  ┌────────────────────┐     │      ┌────────────────┐
│  │ Model Training     │─────┼─────│  Model Registry│
│  └────────┬───────────┘     │      └────────────────┘
│           │                 │
│           ▼                 │
│  ┌────────────────────┐     │
│  │ Model Evaluation   │     │
│  └────────────────────┘     │
└─────────────────────────────┘
```

### Step 4: Component Design (15-20 min)

Deep dive into key components:
- Data pipeline
- Feature engineering
- Model training
- Model serving
- Monitoring and retraining

### Step 5: Trade-offs and Optimization (10 min)

Discuss:
- Accuracy vs. latency
- Complexity vs. interpretability
- Cost vs. performance
- Freshness vs. stability

## Common ML System Patterns

### 1. Recommendation System

**Use Cases:** E-commerce, content platforms, social media

**Architecture:**
```
Two-Stage Ranking:

Stage 1: Candidate Generation
- Retrieve ~1000 candidates from millions
- Fast, recall-focused
- Methods: Collaborative filtering, ANN search

Stage 2: Ranking
- Rank top candidates precisely
- Slower, precision-focused
- Methods: Gradient boosting, deep learning

Serving:
- Candidate cache (precomputed daily)
- Real-time ranking (<100ms)
- Feature store for user/item features
```

**Key Considerations:**
- Cold start: New users/items
- Diversity: Avoid filter bubble
- Freshness: Balance exploration/exploitation
- Personalization: Privacy vs. accuracy

### 2. Search Ranking

**Use Cases:** Web search, e-commerce search, job search

**Architecture:**
```
Query → Query Understanding → Retrieval → Ranking → Blending → Results

Components:
1. Query Understanding: Intent, entities, spelling
2. Retrieval: Boolean search, semantic search
3. Ranking: Learning to rank (LambdaMART, deep learning)
4. Blending: Diversity, personalization
```

**Key Considerations:**
- Query latency (<200ms)
- Index freshness
- Relevance vs. diversity
- Click position bias

### 3. Fraud Detection

**Use Cases:** Payment fraud, account takeover, fake accounts

**Architecture:**
```
Transaction → Feature Extraction → Real-time Scoring → Decision

Layers:
1. Rule-based filters (block obvious fraud)
2. ML model (score suspicious transactions)
3. Human review (for borderline cases)
```

**Key Considerations:**
- Low latency (<50ms for payment decisions)
- Class imbalance (0.1% fraud rate)
- Adversarial: Fraudsters adapt
- False positive cost: Blocking legitimate users

### 4. Content Moderation

**Use Cases:** Social media, forums, marketplaces

**Architecture:**
```
Content → Multiple Classifiers → Ensemble → Decision

Classifiers:
- Text: Toxic language, spam
- Images: NSFW, violence
- User: Reputation score
- Context: Time, location, frequency

Decision:
- Auto-approve (high confidence safe)
- Auto-reject (high confidence unsafe)
- Human review (borderline)
```

**Key Considerations:**
- Near-real-time (<1s)
- Multi-modal (text, images, video)
- Evolving: New abuse types emerge
- Cultural context: Language, geography

### 5. Personalized Feed Ranking

**Use Cases:** Social media feeds, news aggregators

**Architecture:**
```
User → Candidate Generation → Ranking → Diversification → Feed

Candidate Sources:
- Following (social graph)
- Interests (content-based)
- Trending (popularity)

Ranking:
- Engagement prediction (likes, comments, shares)
- Dwell time prediction
- Negative signals (hide, report)

Diversification:
- Topic diversity
- Source diversity
- Freshness
```

**Key Considerations:**
- Real-time updates
- User engagement vs. well-being
- Filter bubble vs. discovery
- A/B testing methodology

## Component Deep Dives

### Data Pipeline

**Batch Processing:**
```
Data Sources → Extract → Transform → Load → Storage

Tools: Apache Spark, Apache Airflow, dbt

Use Cases:
- Training data preparation
- Feature computation
- Model retraining

Considerations:
- Scalability: Petabyte-scale data
- Reliability: Retry logic, monitoring
- Cost: Spot instances, data compression
```

**Stream Processing:**
```
Data Streams → Process → Store/Forward

Tools: Apache Kafka, Apache Flink, Kinesis

Use Cases:
- Real-time features
- Online learning
- Monitoring

Considerations:
- Latency: Sub-second processing
- Fault tolerance: Exactly-once semantics
- Backpressure handling
```

### Feature Store

**Purpose:**
- Centralized feature repository
- Consistency between training and serving
- Feature reusability and discovery

**Architecture:**
```
┌──────────────────────────────────────┐
│           Feature Store              │
│                                      │
│  ┌────────────┐    ┌──────────────┐ │
│  │  Offline   │    │    Online    │ │
│  │  Storage   │    │   Storage    │ │
│  │  (S3/GCS)  │    │  (Redis/     │ │
│  │            │    │   DynamoDB)  │ │
│  └────────────┘    └──────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  Feature Registry/Metadata     │ │
│  │  (Schema, Lineage, Owners)     │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
```

**Key Features:**
- Point-in-time correctness (avoid data leakage)
- Low-latency serving (<10ms)
- Batch and real-time updates
- Monitoring and data quality

**Popular Solutions:**
- Tecton, Feast, AWS SageMaker Feature Store

### Model Training

**Distributed Training:**
```
Strategies:
1. Data Parallelism: Split data across GPUs
2. Model Parallelism: Split model across GPUs
3. Pipeline Parallelism: Split layers across GPUs

Tools:
- PyTorch DDP, Horovod
- TensorFlow Distribution Strategy
- Ray Train
```

**Hyperparameter Tuning:**
```
Methods:
- Grid Search: Exhaustive, expensive
- Random Search: More efficient
- Bayesian Optimization: Sample efficient
- Hyperband: Early stopping

Tools:
- Optuna, Ray Tune, Weights & Biases
```

**Experiment Tracking:**
```
Track:
- Hyperparameters
- Metrics (train/val/test)
- Artifacts (models, plots)
- Code version, data version

Tools:
- MLflow, Weights & Biases, Neptune
```

### Model Serving

**Serving Patterns:**

**1. Model-as-Service**
```
Client → API Gateway → Model Server → Response

Pros:
+ Centralized, easy to update
+ Resource efficient
+ A/B testing

Cons:
- Network latency
- Single point of failure
- Requires internet

Tools: TensorFlow Serving, TorchServe, FastAPI
```

**2. Embedded Model**
```
Client (with embedded model) → Prediction

Pros:
+ No network latency
+ Works offline
+ Privacy (no data sent)

Cons:
- Large binary size
- Harder to update
- Device heterogeneity

Use Cases: Mobile apps, edge devices
```

**3. Hybrid**
```
Client → Edge Model → (Fallback) → Cloud Model

Pros:
+ Low latency for common cases
+ Fallback for complex cases

Use Cases: Voice assistants, mobile apps
```

**Optimization Techniques:**

**Model Compression:**
- Quantization: FP32 → INT8 (4x smaller, faster)
- Pruning: Remove unimportant weights
- Knowledge Distillation: Large model → Small model
- Architectures: MobileNet, EfficientNet

**Serving Optimizations:**
- Batching: Group requests for throughput
- Caching: Cache frequent predictions
- Model sharding: Distribute large models
- GPU/TPU inference: For deep learning

**Latency Targets:**
```
< 10ms: Embedded systems
< 100ms: Real-time services (search, ads)
< 1s: Near-real-time (fraud detection)
< 10s: Batch-like (recommendations precomputed)
```

### Monitoring and Retraining

**Monitoring:**

**Model Performance:**
- Prediction distribution
- Confidence scores
- Accuracy metrics (if labels available)

**Data Quality:**
- Feature distribution shifts
- Missing values
- Outliers

**System Health:**
- Latency (p50, p95, p99)
- Throughput (QPS)
- Error rates
- Resource utilization (CPU, memory, GPU)

**Business Metrics:**
- Revenue, engagement, satisfaction
- A/B test results

**Tools:**
- Prometheus + Grafana
- Datadog, New Relic
- Evidently AI, Whylabs

**Retraining Strategy:**

**Triggers:**
1. **Scheduled**: Weekly, monthly
2. **Performance**: Metrics drop below threshold
3. **Data Drift**: Significant distribution shift
4. **Concept Drift**: Relationship X→Y changes

**Process:**
```
1. Detect need for retraining
2. Collect new labeled data
3. Retrain model (incremental or from scratch)
4. Evaluate on holdout set
5. A/B test in production
6. Gradual rollout (1% → 10% → 100%)
7. Monitor and iterate
```

**Incremental vs. Full Retraining:**
- Incremental: Update with new data (faster, cheaper)
- Full: Retrain from scratch (better for drift)

## Case Studies

### Case Study 1: YouTube Recommendation System

**Requirements:**
- Recommend videos to 2B users
- Real-time recommendations (<100ms)
- Balance engagement with user well-being
- Cold start for new users/videos

**Architecture:**
```
Two-Stage Ranking:

Candidate Generation:
- User history → Embed → ANN Search → 1000 candidates
- Sources: Subscriptions, watch history, trending

Ranking:
- Deep neural network
- Features: User, video, context
- Objective: Engagement (watch time)

Diversification:
- Filter duplicate channels
- Topic diversity
- Freshness boost
```

**Key Design Decisions:**
- Multi-task learning: Engagement + satisfaction
- Sample negative examples: Not watched videos
- Feature freshness: Real-time user activity
- A/B testing framework: Rigorous experimentation

**Scale:**
- Billions of predictions per day
- Millions of videos
- Real-time feature updates

### Case Study 2: Uber ETA Prediction

**Requirements:**
- Predict arrival time for rides
- Sub-second latency
- Accuracy within 1-2 minutes
- Handle traffic, weather, events

**Architecture:**
```
Request → Feature Extraction → Model Inference → ETA

Features:
- Route: Distance, historical duration
- Real-time: Current traffic, weather
- Driver: Speed, behavior
- Time: Hour, day of week, holidays
```

**Model:**
- Gradient Boosting (XGBoost)
- Ensemble: Combine multiple models
- Online learning: Update with recent rides

**Challenges:**
- Data drift: Traffic patterns change
- Sparse data: New routes, rare events
- Cold start: New cities
- Latency: Sub-second requirement

**Solutions:**
- Fallback to distance-based ETA if model unavailable
- Periodic retraining (daily)
- Feature caching for low latency
- Monitor accuracy per city, time of day

### Case Study 3: LinkedIn Feed Ranking

**Requirements:**
- Rank posts in user feed
- Optimize for engagement and time well spent
- Handle billions of posts daily
- Real-time updates

**Architecture:**
```
Two-Stage Ranking:

Candidate Generation:
- Following graph (1st/2nd degree)
- Interests (hashtags, companies)
- Trending posts
- → 500 candidates

Ranking:
- Multi-task neural network
- Predictions: Like, comment, share, hide
- Features: User, post, author, context

Diversification:
- Filter duplicates
- Topic variety
- Connection strength
```

**Key Features:**
- User: Job, industry, interests
- Post: Text, images, author
- Author: Followers, activity
- Context: Time, device

**Training:**
- Positive: User engaged (like, comment)
- Negative: Shown but not engaged
- Sample weight: Negative downsampling
- Update: Daily with latest data

## Interview Tips

### Do's

1. **Clarify Requirements**
   - Ask about scale, latency, accuracy
   - Understand trade-offs

2. **Start High-Level**
   - Draw architecture diagram
   - Explain data flow
   - Then drill into components

3. **Discuss Trade-offs**
   - Every decision has pros/cons
   - "We could use X, but Y is better because..."

4. **Consider Production**
   - Monitoring, retraining, deployment
   - Not just training accuracy

5. **Think End-to-End**
   - Data → Model → Serving → Monitoring

6. **Use Real Examples**
   - Reference systems you've built or know
   - Show practical experience

### Don'ts

1. **Don't Jump to Details**
   - Start with requirements and high-level design
   - Drill down when asked

2. **Don't Ignore Scale**
   - 1K vs 1M vs 1B users matters
   - Design accordingly

3. **Don't Over-engineer**
   - Simple solutions often work
   - Add complexity only when needed

4. **Don't Forget Data**
   - ML systems live and die by data
   - Discuss data quality, labeling, drift

5. **Don't Ignore Operations**
   - How do you monitor?
   - How do you retrain?
   - How do you debug?

### Communication Tips

1. **Think Out Loud**
   - Explain your reasoning
   - Walk through trade-offs

2. **Structure Your Answer**
   - Use the framework above
   - Clear sections

3. **Draw Diagrams**
   - Visual > verbal for architecture
   - Label clearly

4. **Ask Clarifying Questions**
   - Don't assume
   - Engage with interviewer

5. **Summarize**
   - Recap key design decisions
   - Mention alternatives considered

## Summary

ML system design requires balancing multiple concerns:
- **Accuracy**: Model performance
- **Latency**: Response time
- **Scale**: Handling load
- **Cost**: Compute and storage
- **Reliability**: Uptime and correctness
- **Maintainability**: Monitoring and retraining

The key is to:
1. Understand requirements and constraints
2. Make explicit trade-offs
3. Think end-to-end (data to deployment)
4. Consider production realities

Practice with the case studies in this repository to build intuition for these trade-offs!
