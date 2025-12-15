# Problem Statement: End-to-End ML Pipeline for Customer Churn Prediction

## Learning Objectives

This case study demonstrates how to design and implement a complete machine learning pipeline from data ingestion to model serving and monitoring. It's designed for **interview preparation** and covers concepts frequently asked in ML System Design interviews at FAANG+ companies.

### Key Concepts Covered
- Data ingestion and validation
- Feature engineering and feature stores
- Model training and experiment tracking
- Model evaluation and selection
- Model serving (real-time and batch)
- Monitoring and drift detection
- MLOps best practices
- System architecture and scalability

---

## Business Context

### Scenario
You're building an ML system for a **subscription-based streaming service** (similar to Netflix, Spotify, or SaaS platforms) that wants to predict which customers are likely to cancel their subscriptions (churn) in the next 30 days.

### Why This Problem Matters
- **High business impact**: Acquiring new customers costs 5-7x more than retaining existing ones
- **Proactive intervention**: Identifying at-risk customers enables targeted retention campaigns
- **Revenue protection**: Early prediction allows time for personalized offers or engagement
- **ROI quantification**: Clear business metrics make it easy to measure ML impact

### Real-World Examples
- **Netflix**: Predicts churn to personalize content recommendations
- **Spotify**: Identifies users likely to downgrade from Premium to Free tier
- **SaaS companies**: Target high-value customers at risk of canceling
- **Telecom**: Reduce subscriber churn through proactive customer service

---

## Problem Definition

### Task Type
**Binary Classification**: Predict whether a customer will churn (1) or remain (0) in the next 30 days

### Input Data
Customer data includes:
- **Demographics**: Age, gender, location, signup date
- **Subscription info**: Plan type, payment method, tenure
- **Usage patterns**: Login frequency, features used, session duration, content consumed
- **Engagement metrics**: Customer support interactions, app ratings, social engagement
- **Transaction history**: Payment delays, plan changes, billing issues

### Success Metrics

#### Model Metrics
- **Recall (Primary)**: Minimize false negatives (missing at-risk customers is costly)
- **Precision**: Avoid false positives (retention campaigns have costs)
- **ROC-AUC**: Overall model discrimination ability
- **F1-Score**: Balance between precision and recall

#### Business Metrics
- **Customer Lifetime Value (CLV) saved**: Revenue retained through interventions
- **Retention campaign ROI**: Value saved vs campaign costs
- **Churn rate reduction**: Overall decrease in monthly churn %
- **Intervention success rate**: % of predicted churners who stay after campaign

---

## Requirements

### Functional Requirements

1. **Data Ingestion**
   - Ingest customer data from multiple sources (databases, event streams, APIs)
   - Validate data quality and schema compliance
   - Handle missing values and outliers
   - Support both batch and streaming ingestion

2. **Feature Engineering**
   - Transform raw data into predictive features
   - Create time-based aggregations (30-day, 90-day windows)
   - Generate behavioral features (usage trends, engagement scores)
   - Ensure training/serving consistency

3. **Model Training**
   - Train multiple model types (Logistic Regression, Random Forest, XGBoost)
   - Handle class imbalance (~85% non-churn, ~15% churn)
   - Perform hyperparameter tuning
   - Track experiments and model versions
   - Use time-based cross-validation (avoid data leakage)

4. **Model Evaluation**
   - Calculate multiple performance metrics
   - Perform threshold tuning for business objectives
   - Compare model versions
   - Generate evaluation reports

5. **Model Serving**
   - **Real-time predictions**: REST API for on-demand scoring (<100ms latency)
   - **Batch predictions**: Score entire customer base daily
   - Model versioning and rollback capability
   - A/B testing support

6. **Monitoring & Alerting**
   - Detect data drift (feature distribution changes)
   - Monitor model performance degradation
   - Track prediction latency and throughput
   - Alert on anomalies
   - Trigger retraining when needed

### Non-Functional Requirements

#### Performance
- **Latency**: <100ms for real-time predictions (p99)
- **Throughput**: Handle 1,000+ predictions/second
- **Batch scoring**: Process 1M customers in <1 hour

#### Scalability
- Support growing customer base (10M+ customers)
- Handle increasing feature complexity
- Scale horizontally for serving
- Distributed training for large datasets

#### Reliability
- **Availability**: 99.9% uptime for prediction service
- **Fault tolerance**: Graceful degradation if model unavailable
- **Disaster recovery**: Model backup and rollback capabilities

#### Maintainability
- Clear code documentation and architecture
- Automated testing (unit, integration, end-to-end)
- CI/CD pipeline for model deployment
- Version control for code, data, and models

---

## Data Characteristics

### Dataset Size
- **Training set**: 100,000 customers (historical data, 6 months)
- **Features**: ~30 features after engineering
- **Class distribution**: Imbalanced (85% non-churn, 15% churn)

### Temporal Aspects
- **Historical data**: 12 months of customer activity
- **Prediction window**: Next 30 days
- **Training window**: 6 months of historical data
- **Feature lookback**: Varying windows (7, 30, 90 days)

### Data Quality Challenges
- **Missing values**: ~5-10% in optional fields
- **Outliers**: Extreme usage patterns (power users, dormant accounts)
- **Delayed events**: Logging delays can affect real-time features
- **Schema evolution**: New features added over time

---

## Challenges & Considerations

### 1. Class Imbalance
**Problem**: Only 15% of customers churn (imbalanced dataset)

**Solutions to Discuss**:
- SMOTE (Synthetic Minority Over-sampling)
- Class weights in model training
- Threshold tuning for business objectives
- Ensemble methods
- Cost-sensitive learning

**Interview Tip**: Discuss why accuracy is a poor metric for imbalanced data

### 2. Temporal Data Leakage
**Problem**: Using future information in training leads to overly optimistic results

**Solutions**:
- Time-based train/test splits (no random shuffling)
- Rolling window validation
- Careful feature engineering (only use past data)
- Clear temporal boundaries

**Interview Tip**: Explain how data leakage happens and how to prevent it

### 3. Training/Serving Skew
**Problem**: Features computed differently in training vs production

**Solutions**:
- Feature store for consistency
- Same preprocessing pipeline for training and serving
- Feature validation checks
- Integration testing

**Interview Tip**: Discuss feature store architecture and benefits

### 4. Concept Drift
**Problem**: Customer behavior changes over time (seasonality, trends, external events)

**Solutions**:
- Continuous monitoring of feature distributions
- Performance tracking over time
- Automated retraining pipelines
- Ensemble of models (recent + historical)

**Interview Tip**: Explain data drift vs concept drift vs model drift

### 5. Threshold Selection
**Problem**: Default 0.5 threshold may not optimize business objectives

**Solutions**:
- Cost-benefit analysis for false positives vs false negatives
- Precision-recall curve analysis
- Business-driven threshold selection
- Different thresholds for different customer segments

**Interview Tip**: Connect model outputs to business decisions

### 6. Scalability
**Problem**: System must scale with growing customer base and features

**Solutions**:
- Feature selection to reduce dimensionality
- Model caching and batch predictions
- Horizontal scaling for serving
- Feature computation optimization
- Approximate nearest neighbor for similar patterns

**Interview Tip**: Discuss scaling strategies for each pipeline component

---

## System Constraints

### Infrastructure
- **Compute**: Limited GPU resources for training
- **Storage**: Cost considerations for feature storage
- **Latency**: Network latency for distributed systems

### Business
- **Budget**: Retention campaign costs ~$10 per customer
- **CLV**: Average customer value ~$600/year
- **Churn cost**: Acquiring replacement customer ~$100
- **Campaign capacity**: Can only contact 1,000 customers/day

### Regulatory
- **Privacy**: GDPR compliance for customer data
- **Explainability**: Must explain predictions to customer service teams
- **Fairness**: Avoid bias against demographic groups
- **Data retention**: Delete customer data after cancellation

---

## Expected System Components

This end-to-end pipeline will include:

1. **Data Ingestion Module**: Load and validate customer data
2. **Feature Engineering Pipeline**: Transform raw data into features
3. **Training Pipeline**: Train models with experiment tracking
4. **Evaluation Module**: Assess model performance
5. **Inference Service**: REST API for predictions
6. **Monitoring System**: Track drift and performance
7. **Pipeline Orchestrator**: Coordinate end-to-end workflow

---

## Interview Discussion Points

When presenting this system in an interview, be prepared to discuss:

### System Design
- How do components interact?
- What technologies would you use for each component?
- How do you ensure fault tolerance?
- Where are potential bottlenecks?

### Scalability
- How does the system scale to 100M customers?
- What are the compute vs storage trade-offs?
- When do you need distributed training?
- How do you optimize serving latency?

### ML-Specific
- Why this model type vs others?
- How do you handle class imbalance?
- What features are most predictive?
- How do you prevent overfitting?

### Production
- How do you deploy model updates safely?
- What monitoring is essential?
- How do you debug prediction issues?
- What's the retraining strategy?

### Business Alignment
- How do you measure ML impact?
- What threshold maximizes ROI?
- How do you prioritize which customers to contact?
- What happens if the model is wrong?

---

## Success Criteria

### For Interview Preparation
- [ ] Understand each pipeline component's role
- [ ] Explain design decisions and trade-offs
- [ ] Discuss scalability challenges and solutions
- [ ] Connect ML metrics to business outcomes
- [ ] Demonstrate production ML best practices

### For Technical Implementation
- [ ] Working data generation script
- [ ] Functional pipeline with all components
- [ ] Model achieves recall >0.75, precision >0.60
- [ ] Inference API with <100ms latency
- [ ] Drift detection catches distribution changes
- [ ] Comprehensive test coverage
- [ ] Clear documentation and examples

---

## Next Steps

1. **Review Solution Approach**: Understand the system architecture
2. **Explore Code**: Study each module's implementation
3. **Run Notebooks**: Execute examples to see the pipeline in action
4. **Practice Explaining**: Articulate design decisions out loud
5. **Review Interview Questions**: Prepare for common questions
6. **Understand Trade-offs**: Know when to use different approaches

This problem statement provides the foundation for building a production-grade ML pipeline. The focus is on understanding **why** design decisions are made, not just **how** to implement them.
