# End-to-End ML Pipeline: Customer Churn Prediction

> **Complete ML System Design**: From data ingestion to monitoring in production

## Table of Contents
- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Quick Start](#quick-start)
- [What You'll Build](#what-youll-build)
- [Project Structure](#project-structure)
- [Key Concepts Covered](#key-concepts-covered)
- [Interview Preparation](#interview-preparation)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Success Metrics](#success-metrics)
- [Common Pitfalls](#common-pitfalls)
- [Next Steps](#next-steps)

---

## Overview

**Business Problem**: Predict customer churn for a subscription-based streaming service (Netflix/Spotify-style) to enable proactive retention strategies.

**ML Problem**: Design and implement a complete ML pipeline including:
- Data ingestion & feature engineering
- Model training with experiment tracking
- Model serving via API
- Real-time monitoring & drift detection
- Automated retraining

**Difficulty**: 🔴 Advanced

**Time to Complete**: 7-9 hours (spread over multiple sessions recommended)

**Key Skills**:
- ML System Architecture
- Feature Engineering & Feature Stores
- Model Training & Hyperparameter Tuning
- API Development (FastAPI)
- Experiment Tracking (MLflow)
- Monitoring & Observability
- Class Imbalance Handling
- Production Best Practices

---

## Learning Objectives

### After completing this case study, you will be able to:

1. **System Design**
   - Design end-to-end ML architectures for production
   - Make informed trade-offs between complexity, cost, and performance
   - Explain data flow from ingestion to prediction

2. **Feature Engineering**
   - Design feature stores to prevent training/serving skew
   - Implement point-in-time correct features
   - Create derived features that encode domain knowledge

3. **Model Training**
   - Handle class imbalance (15% churn rate)
   - Use experiment tracking (MLflow) to compare models
   - Optimize hyperparameters using Bayesian search (Optuna)
   - Select models based on business metrics, not just accuracy

4. **Model Serving**
   - Build production APIs using FastAPI
   - Implement batch and real-time prediction modes
   - Handle edge cases and validation

5. **Monitoring**
   - Detect data drift using statistical tests (KS test, PSI)
   - Monitor concept drift (model performance degradation)
   - Set up automated retraining triggers

6. **Interview Skills**
   - Answer ML system design questions confidently
   - Discuss trade-offs with concrete examples
   - Connect technical decisions to business outcomes

---

## Quick Start

### Option 1: Explore the Documentation (Recommended for Interview Prep)
Start by reading the comprehensive documentation to understand concepts:

```bash
# 1. Read the problem statement
open problem_statement.md  # Or use your preferred text editor

# 2. Study the solution architecture
open solution_approach.md

# 3. Review common interview questions
open interview_questions.md

# 4. Understand design trade-offs
open trade_offs.md
```

### Option 2: Hands-On Implementation
Work with the dataset and build models:

```bash
# 1. Navigate to the directory
cd 07_ml-system-design/01_end-to-end-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic dataset (10,000 customers)
python3 data/generate_data.py

# 4. Explore the data
jupyter notebook notebooks/01_eda.ipynb

# 5. Build and train models
jupyter notebook notebooks/02_model_training.ipynb

# 6. Start the prediction API
python3 src/api.py

# 7. Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST_001", "logins_last_30d": 2, ...}'
```

---

## What You'll Build

### 1. Data Pipeline
- **Data Ingestion**: Load customer data from databases/files
- **Feature Engineering**: Transform raw data into ML-ready features
- **Feature Store**: Cache features for training and serving

### 2. Training Pipeline
- **Data Preprocessing**: Handle categorical variables, scaling
- **Class Imbalance**: SMOTE, class weights, threshold tuning
- **Model Selection**: Compare Logistic Regression, Random Forest, XGBoost
- **Hyperparameter Tuning**: Optimize using Optuna
- **Experiment Tracking**: Log metrics, parameters, artifacts with MLflow

### 3. Serving Pipeline
- **Batch Predictions**: Daily scoring of all customers
- **Real-time API**: On-demand predictions for high-value customers
- **A/B Testing**: Canary deployment for new models
- **Caching**: Feature cache to reduce latency

### 4. Monitoring Pipeline
- **Data Quality**: Validate input features
- **Data Drift**: Detect distribution changes (Evidently)
- **Concept Drift**: Monitor precision/recall degradation
- **Alerting**: Trigger retraining when performance drops

### 5. MLOps Infrastructure
- **Model Registry**: Version and manage models (MLflow)
- **CI/CD**: Automated testing and deployment
- **Logging**: Structured logs for debugging
- **Dashboards**: Grafana/Prometheus for system monitoring

---

## Project Structure

```
01_end-to-end-pipeline/
├── README.md                      # This file - learning objectives & quick start
├── problem_statement.md            # Business context & requirements (15 KB)
├── solution_approach.md            # System architecture & design (48 KB)
├── interview_questions.md          # 20+ Q&As for interviews (85 KB)
├── trade_offs.md                   # Design decisions & trade-offs (75 KB)
├── requirements.txt                # Python dependencies with comments
│
├── data/                           # Dataset & data documentation
│   ├── customers.csv               # 10,000 customer records (1.5 MB)
│   ├── customers.json              # JSON format (9.3 MB)
│   ├── dataset_summary.txt         # Quick stats reference
│   ├── data_description.md         # Schema & feature descriptions
│   └── generate_data.py            # Data generation script
│
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb               # Exploratory data analysis
│   ├── 02_model_training.ipynb    # Model development & tuning
│   ├── 03_threshold_tuning.ipynb  # Business-driven threshold optimization
│   ├── 04_monitoring.ipynb        # Drift detection & monitoring
│   └── 05_deployment.ipynb        # API testing & deployment
│
├── src/                            # Production Python modules
│   ├── __init__.py
│   ├── data_ingestion.py          # Load data from sources
│   ├── feature_engineering.py     # Feature store implementation
│   ├── model_training.py          # Training pipeline
│   ├── model_serving.py           # Prediction service
│   ├── api.py                     # FastAPI application
│   ├── monitoring.py              # Drift detection
│   └── utils.py                   # Helper functions
│
├── tests/                          # Unit tests (pytest)
│   ├── test_feature_engineering.py
│   ├── test_model_training.py
│   ├── test_api.py
│   └── conftest.py                # Pytest fixtures
│
└── configs/                        # Configuration files
    ├── training_config.yaml       # Model training settings
    ├── serving_config.yaml        # API configuration
    └── monitoring_config.yaml     # Drift detection thresholds
```

**Total Documentation**: ~235 KB of interview-focused content

---

## Key Concepts Covered

### 1. ML System Design (Architecture)
- **Component breakdown**: Data, training, serving, monitoring
- **Data flow**: Request → Feature Store → Model → Prediction
- **Scaling patterns**: Batch vs. streaming, horizontal scaling
- **Cost optimization**: Caching, model compression, sampling

### 2. Data & Feature Engineering
- **Feature Store**: Custom implementation vs. Feast vs. managed services
- **Point-in-time correctness**: Avoid data leakage with temporal features
- **Derived features**: Engagement scores, trend detection, risk flags
- **Training/Serving consistency**: Same logic for both pipelines

### 3. Model Training & Evaluation
- **Class imbalance**: 15% churn rate (realistic)
  - SMOTE (synthetic oversampling)
  - Class weights (balanced)
  - Threshold tuning (business-driven)
- **Metrics**: Precision, recall, F1, ROC-AUC, **business ROI**
- **Hyperparameter tuning**: Grid search, random search, Bayesian optimization
- **Experiment tracking**: MLflow for reproducibility

### 4. Model Serving & Deployment
- **API design**: RESTful endpoints with FastAPI
- **Batch predictions**: Daily scoring for all customers
- **Real-time predictions**: <100ms latency for interactive use cases
- **Deployment strategies**: Blue-green, canary, shadow mode
- **Error handling**: Validation, timeouts, fallback to defaults

### 5. Monitoring & Operations
- **Data drift**: KS test, PSI (Population Stability Index)
- **Concept drift**: Monitor precision/recall over time
- **Retraining triggers**: Performance drops, distribution changes
- **Alerting**: Slack/PagerDuty integration
- **Sampled logging**: 1-5% of predictions for analysis

### 6. Production Best Practices
- **Testing**: Unit tests, integration tests, API tests
- **Logging**: Structured logs with correlation IDs
- **Versioning**: Models, features, code
- **Documentation**: API docs (automatic with FastAPI)
- **Observability**: Metrics (Prometheus), logs (ELK), traces (Jaeger)

---

## Interview Preparation

This case study is designed to help you ace ML system design interviews at FAANG+ companies.

### Top 10 Interview Topics Covered

| # | Topic | Where to Find It | Why It Matters |
|---|-------|-----------------|----------------|
| 1 | System Architecture | [solution_approach.md:Lines 50-200](solution_approach.md#system-architecture) | 90% of interviews start with "Design a churn prediction system" |
| 2 | Class Imbalance | [interview_questions.md:Q8](interview_questions.md#q8-how-do-you-handle-class-imbalance-in-churn-prediction) | Most real-world datasets are imbalanced |
| 3 | Feature Store | [solution_approach.md:Component 2](solution_approach.md#2-feature-engineering--feature-store) | Prevents training/serving skew (common bug!) |
| 4 | Monitoring & Drift | [interview_questions.md:Q14-16](interview_questions.md#q14-how-do-you-monitor-model-performance-in-production) | Production reliability is critical |
| 5 | Batch vs Real-time | [trade_offs.md:Trade-off 1](trade_offs.md#1-real-time-vs-batch-predictions) | Cost vs. latency trade-off |
| 6 | Model Selection | [interview_questions.md:Q7](interview_questions.md#q7-how-do-you-select-the-best-model) | XGBoost vs. deep learning vs. simple models |
| 7 | Threshold Tuning | [interview_questions.md:Q8](interview_questions.md#q8-how-do-you-handle-class-imbalance-in-churn-prediction) | Business-driven decision boundaries |
| 8 | Scalability | [interview_questions.md:Q17-18](interview_questions.md#q17-how-do-you-scale-the-system-to-handle-10x-traffic) | From 1M to 100M customers |
| 9 | A/B Testing | [solution_approach.md:Component 4](solution_approach.md#4-model-serving-deployment) | Canary deployments, shadow mode |
| 10 | Business Metrics | [interview_questions.md:Q19-20](interview_questions.md#q19-how-do-you-measure-the-business-impact-of-the-churn-model) | ROI, CLV, cost-benefit analysis |

### Recommended Study Path

**Phase 1: Understand the Problem (1-2 hours)**
1. Read [problem_statement.md](problem_statement.md) - Business context
2. Review [data/data_description.md](data/data_description.md) - Dataset schema
3. Generate and explore the dataset

**Phase 2: Learn the Architecture (2-3 hours)**
1. Study [solution_approach.md](solution_approach.md) - System design
2. Understand each component (data, training, serving, monitoring)
3. Review code examples embedded in documentation

**Phase 3: Master Trade-offs (1-2 hours)**
1. Read [trade_offs.md](trade_offs.md) - 10 major design decisions
2. Memorize pros/cons of each option
3. Practice explaining trade-offs out loud

**Phase 4: Practice Interview Questions (2-3 hours)**
1. Read [interview_questions.md](interview_questions.md) - 20+ detailed Q&As
2. Practice answering without looking at answers
3. Focus on Top 10 questions (see table above)

**Phase 5: Hands-On Coding (Optional, 3-5 hours)**
1. Implement key components (feature store, API)
2. Train models and compare metrics
3. Set up monitoring dashboard

---

## Prerequisites

### Required Knowledge
- **Python**: Intermediate (functions, classes, list comprehensions)
- **Machine Learning**: Supervised learning basics (classification, metrics)
- **APIs**: Basic understanding of REST APIs (GET, POST)
- **SQL**: Basic queries (SELECT, JOIN, WHERE)

### Recommended (Not Required)
- **Docker**: For containerization (can skip initially)
- **Cloud Platforms**: AWS/GCP/Azure familiarity (concepts translate)
- **Distributed Systems**: Helpful but not essential

### Technical Setup
- **Python**: 3.8+ (3.9+ recommended)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk Space**: 5 GB (for dependencies and datasets)
- **OS**: macOS, Linux, or Windows (with WSL)

---

## Getting Started

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd 07_ml-system-design/01_end-to-end-pipeline

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import sklearn, xgboost, fastapi; print('✓ All dependencies installed')"
```

### Step 2: Generate Dataset

```bash
# Generate 10,000 synthetic customer records
python3 data/generate_data.py

# Verify dataset
ls -lh data/customers.csv  # Should be ~1.5 MB

# Quick preview
head -n 10 data/customers.csv
```

**Output**: `data/customers.csv`, `data/customers.json`, `data/dataset_summary.txt`

### Step 3: Explore Documentation

```bash
# Read the problem statement (business context)
cat problem_statement.md | less

# Study the solution architecture
cat solution_approach.md | less

# Review interview questions (with answers!)
cat interview_questions.md | less
```

### Step 4: Run Notebooks (Optional)

```bash
# Start Jupyter notebook server
jupyter notebook

# Open notebooks in this order:
# 1. notebooks/01_eda.ipynb          - Explore data
# 2. notebooks/02_model_training.ipynb - Train models
# 3. notebooks/03_threshold_tuning.ipynb - Optimize threshold
```

### Step 5: Build & Test API (Optional)

```bash
# Train a model first (saves to mlruns/)
python3 src/model_training.py

# Start the FastAPI server
python3 src/api.py

# In another terminal, test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "tenure_days": 120,
    "logins_last_30d": 2,
    "satisfaction_score": 2.5,
    "payment_failures_6m": 3,
    "auto_renew_enabled": 0
  }'

# Response: {"customer_id": "CUST_001", "churn_probability": 0.85, "risk_level": "HIGH"}
```

---

## Documentation

### Core Documents (Read in Order)

1. **[problem_statement.md](problem_statement.md)** (~15 KB)
   - Business context and requirements
   - Success metrics (recall >0.75, precision >0.60)
   - 6 key challenges (class imbalance, data leakage, drift, etc.)

2. **[solution_approach.md](solution_approach.md)** (~48 KB)
   - System architecture diagram
   - 6 component deep-dives with code examples
   - Technology stack recommendations

3. **[interview_questions.md](interview_questions.md)** (~85 KB)
   - 20+ detailed Q&As across 7 categories
   - Code examples and diagrams
   - Top 10 interview tips + preparation checklist

4. **[trade_offs.md](trade_offs.md)** (~75 KB)
   - 10 major design decisions analyzed
   - Pros/cons/recommendations with cost estimates
   - Quick reference table

5. **[data/data_description.md](data/data_description.md)** (~12 KB)
   - Dataset schema and feature descriptions
   - Expected feature importance
   - Usage guidelines for ML pipeline

### Supplementary Documents

- **[requirements.txt](requirements.txt)**: Dependencies with interview-focused comments
- **data/dataset_summary.txt**: Quick dataset statistics
- **configs/**: YAML configuration examples (to be created)

---

## Success Metrics

### Technical Metrics (Model Performance)
- **ROC-AUC**: >0.85 (excellent discrimination)
- **Recall**: >0.75 (catch 75% of churners)
- **Precision**: >0.60 (avoid too many false alarms)
- **API Latency**: <100ms p99 (real-time predictions)

### Business Metrics (Impact)
- **Retention Rate**: +2-5% improvement
- **ROI**: >300% (saved CLV vs. retention costs)
- **Cost Savings**: $600 CLV * churners saved - $10 * offers sent
- **Reactivation Rate**: 20-30% of targeted customers retained

### Learning Metrics (Interview Readiness)
- Can you design the system architecture in 15 minutes? ✅
- Can you explain 5+ design trade-offs with pros/cons? ✅
- Can you write pseudo-code for key components? ✅
- Can you calculate business ROI on the spot? ✅
- Can you answer follow-up questions confidently? ✅

---

## Common Pitfalls

### Data & Features
❌ **Using future information** (data leakage)
   - Don't use `days_until_renewal` for long-term predictions
   - Ensure point-in-time correctness

❌ **Training/Serving skew**
   - Feature calculation differs between training and serving
   - Solution: Use feature store with versioned logic

❌ **Ignoring class imbalance**
   - Using accuracy as the primary metric (85% by predicting all "no churn")
   - Solution: Focus on recall, precision, F1, ROC-AUC

### Model Training
❌ **Random train/test split for temporal data**
   - Causes data leakage (future data in training set)
   - Solution: Use time-based split (`signup_date` cutoff)

❌ **Optimizing for wrong metric**
   - Maximizing accuracy instead of business ROI
   - Solution: Tune threshold based on CLV and retention costs

❌ **Not tracking experiments**
   - Forgetting what hyperparameters were used
   - Solution: Use MLflow to log all experiments

### Deployment
❌ **No monitoring**
   - Model degrades silently in production
   - Solution: Data drift detection, performance tracking

❌ **No fallback**
   - API returns 500 when model fails
   - Solution: Return default (e.g., average churn rate) on errors

❌ **Deploying to 100% of traffic immediately**
   - Bug affects all users
   - Solution: Canary deployment (5% → 25% → 100%)

---

## Next Steps

### After Completing This Case Study

1. **Implement Advanced Features**
   - Add streaming features (Kafka + real-time aggregations)
   - Implement online learning (update model with new data)
   - Build explainability dashboard (SHAP values for each prediction)

2. **Explore Related Case Studies**
   - [Fraud Detection](../../02_classification/01_fraud-detection/) - Similar class imbalance challenges
   - [Recommendation Systems](../../01_recommendation-systems/) - Personalization techniques
   - [Text-to-SQL](../../08_generative-ai-llms/01_text-to-sql/) - Another production ML system

3. **Practice Interview Questions**
   - Mock interviews with peers using [interview_questions.md](interview_questions.md)
   - Draw system architecture diagrams on whiteboard
   - Time yourself: 45 minutes for full system design

4. **Read Production ML Resources**
   - **Book**: "Designing Machine Learning Systems" by Chip Huyen
   - **Course**: Made With ML (MLOps best practices)
   - **Blog**: Netflix TechBlog (real-world ML systems)

---

## Contributing

Found an issue or have suggestions? See [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## License

This project is part of the AI/ML Interview Prep repository.
See [LICENSE](../../LICENSE) for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-usecases/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-usecases/discussions)
- **Email**: your-email@example.com

---

**Last Updated**: 2025-12-15
**Status**: ✅ Complete - All documentation and data ready
**Version**: 1.0

---

## Quick Links

- [Problem Statement](problem_statement.md) - Start here
- [Solution Architecture](solution_approach.md) - System design
- [Interview Questions](interview_questions.md) - 20+ Q&As
- [Trade-offs](trade_offs.md) - Design decisions
- [Data Description](data/data_description.md) - Dataset schema
- [Requirements](requirements.txt) - Dependencies

**Ready to start?** → [problem_statement.md](problem_statement.md)
