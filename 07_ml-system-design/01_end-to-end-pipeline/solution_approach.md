# Solution Approach: End-to-End ML Pipeline

## Overview

This document describes the architecture and implementation approach for a production ML pipeline to predict customer churn. The design prioritizes **scalability**, **maintainability**, and **production readiness** while covering key concepts for ML system design interviews.

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                   │
│  Customer DB   │   Event Streams   │   Transaction DB   │   External API│
└────────────┬─────────────┬───────────────┬────────────────┬─────────────┘
             │             │               │                 │
             └─────────────┴───────────────┴─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │    DATA INGESTION LAYER        │
                    │  • Batch Ingestion             │
                    │  • Stream Ingestion            │
                    │  • Data Validation             │
                    │  • Schema Enforcement          │
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │   FEATURE ENGINEERING LAYER    │
                    │  • Feature Transformation      │
                    │  • Time-based Aggregations     │
                    │  • Feature Store (Consistency) │
                    │  • Feature Versioning          │
                    └──────────────┬─────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                      │
    ┌───────────▼────────────┐         ┌─────────────▼──────────────┐
    │  TRAINING PIPELINE     │         │   INFERENCE PIPELINE       │
    │ • Model Training       │         │ • Real-time Predictions    │
    │ • Hyperparameter Tuning│         │ • Batch Predictions        │
    │ • Experiment Tracking  │         │ • Model Serving (API)      │
    │ • Model Evaluation     │         │ • Result Storage           │
    │ • Model Registry       │         └──────────┬─────────────────┘
    └───────────┬────────────┘                    │
                │                                  │
                └──────────────┬───────────────────┘
                               │
                ┌──────────────▼─────────────────┐
                │   MONITORING & ALERTING        │
                │  • Data Drift Detection        │
                │  • Model Performance Tracking  │
                │  • Latency & Throughput Metrics│
                │  • Alerting & Retraining       │
                └────────────────────────────────┘
```

---

## Component Design

### 1. Data Ingestion Layer

#### Purpose
Load customer data from various sources, validate quality, and prepare for feature engineering.

#### Components

**A. Batch Data Ingestion**
```python
class BatchDataIngestion:
    """Loads historical customer data in batches"""

    def ingest_from_database(self, query, batch_size):
        """Fetch data from relational database"""

    def ingest_from_files(self, file_paths):
        """Load data from CSV/Parquet files"""

    def validate_schema(self, data):
        """Ensure data matches expected schema"""
```

**Key Interview Points**:
- When to use batch vs streaming?
  - **Batch**: Historical data, daily scoring, model training
  - **Streaming**: Real-time features, immediate predictions, event-driven

**B. Stream Data Ingestion**
```python
class StreamDataIngestion:
    """Processes real-time customer events"""

    def consume_from_kafka(self, topic):
        """Consume events from Kafka stream"""

    def process_event(self, event):
        """Validate and transform event"""
```

**Technologies**:
- **Batch**: SQLAlchemy, pandas, Apache Spark (large scale)
- **Streaming**: Kafka, Kinesis, Pub/Sub

**C. Data Validation**
```python
class DataValidator:
    """Validates data quality and schema compliance"""

    def check_missing_values(self, data, threshold=0.1):
        """Alert if missing values exceed threshold"""

    def detect_outliers(self, data, features):
        """Identify anomalous values"""

    def validate_types(self, data, schema):
        """Ensure correct data types"""
```

**Validation Checks**:
- Schema compliance (expected columns, types)
- Missing value thresholds (<10%)
- Range validation (age 18-100, tenure >= 0)
- Distribution checks (detect data drift)
- Referential integrity (customer_id exists)

#### Interview Questions to Prepare

**Q: How would you handle schema evolution (new columns added)?**
- Use schema versioning
- Backwards-compatible transformations
- Feature flags for gradual rollout
- Default values for missing features

**Q: What happens if data validation fails?**
- Log error details for debugging
- Alert on-call engineer
- Skip bad records (don't crash pipeline)
- Store rejected records for later analysis
- Monitor rejection rate (alert if >1%)

---

### 2. Feature Engineering Layer

#### Purpose
Transform raw customer data into predictive features while ensuring training/serving consistency.

#### Key Concepts

**A. Feature Types**

1. **Demographic Features** (Static)
   - Age, gender, location
   - Preprocessing: One-hot encoding, normalization

2. **Subscription Features** (Slowly changing)
   - Plan type, payment method, tenure (days since signup)
   - Preprocessing: Label encoding, log transformation

3. **Behavioral Features** (Time-based aggregations)
   - `logins_last_7_days`, `logins_last_30_days`, `logins_last_90_days`
   - `avg_session_duration_30d`, `total_content_consumed_30d`
   - Preprocessing: Standardization, handle zeros

4. **Engagement Features** (Derived)
   - `login_frequency_trend` (recent vs historical)
   - `days_since_last_login`, `days_since_last_payment`
   - `support_interaction_count_90d`

5. **Transaction Features** (Aggregated)
   - `payment_failures_last_6_months`
   - `plan_changes_last_year`
   - `billing_amount_change_%`

**B. Feature Engineering Pipeline**

```python
class FeatureEngineer:
    """Transforms raw data into model-ready features"""

    def create_time_aggregations(self, data, windows=[7, 30, 90]):
        """Compute rolling window aggregations"""
        # logins_last_7d, logins_last_30d, etc.

    def create_trend_features(self, data):
        """Calculate temporal trends"""
        # recent_activity / historical_activity

    def create_derived_features(self, data):
        """Generate interaction and ratio features"""
        # avg_session_duration = total_time / num_sessions

    def handle_missing_values(self, data):
        """Impute or flag missing data"""
        # Median for numeric, mode for categorical

    def encode_categorical(self, data, features):
        """One-hot or target encoding"""

    def normalize_features(self, data, method='standard'):
        """Standardize or min-max scale"""
```

**C. Feature Store (Training/Serving Consistency)**

**Problem**: Features computed differently in training vs production leads to poor performance.

**Solution**: Centralized feature store ensures consistency.

```python
class FeatureStore:
    """Manages feature computation and storage"""

    def compute_features(self, raw_data, mode='training'):
        """Compute features using same logic for training and serving"""

    def save_features(self, customer_id, features, version):
        """Store features with versioning"""

    def get_features(self, customer_id, version='latest'):
        """Retrieve features for inference"""

    def get_feature_statistics(self):
        """Get mean/std for monitoring"""
```

**Implementation Options**:
- **Simple**: SQLite or Postgres table
- **Production**: Feast, Tecton, AWS Feature Store
- **In-memory**: Redis for low-latency serving

**D. Handling Time-based Features**

**Critical**: Avoid data leakage by using only past data.

```python
# WRONG: Uses future information
features['churned_after_30_days'] = (df['churn_date'] - df['prediction_date']).dt.days < 30

# CORRECT: Uses only historical data
features['logins_last_30_days'] = df[df['login_date'] < prediction_date].groupby('customer_id')['login_date'].count()
```

#### Interview Questions to Prepare

**Q: How do you ensure training/serving consistency?**
- Use same feature engineering code for both
- Feature store with versioned transformations
- Integration tests comparing training and serving features
- Monitor feature distribution differences

**Q: How do you handle missing values in production?**
- Impute using training set statistics (mean/median)
- Use model-based imputation (k-NN, MICE)
- Create "missing" indicator features
- Design features robust to missing values

**Q: What if a feature is delayed in production?**
- Use last known value (with timestamp check)
- Fall back to default value
- Exclude feature if too stale (flag for monitoring)
- Have redundant features

---

### 3. Training Pipeline

#### Purpose
Train multiple model candidates, track experiments, and select the best model for deployment.

#### Components

**A. Model Training**

```python
class ModelTrainer:
    """Trains classification models with proper validation"""

    def train_multiple_models(self, X_train, y_train):
        """Train Logistic Regression, Random Forest, XGBoost"""
        models = {
            'logistic': LogisticRegression(class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100),
            'xgboost': XGBClassifier(scale_pos_weight=5)  # Handle imbalance
        }

    def time_based_split(self, data, test_size=0.2):
        """Split by time to avoid leakage"""
        # Sort by date, take last 20% as test

    def cross_validate(self, X, y, cv=5):
        """Time-series cross-validation"""
        # Use TimeSeriesSplit, not KFold
```

**Handling Class Imbalance**:

1. **Class Weights**: Penalize false negatives more
   ```python
   class_weight = {0: 1, 1: 5}  # Churn class is 5x more important
   ```

2. **SMOTE**: Oversample minority class
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(sampling_strategy=0.5)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

3. **Threshold Tuning**: Adjust decision threshold (not 0.5)
   ```python
   # Choose threshold that maximizes F1 or business metric
   best_threshold = 0.35  # More sensitive to churn
   predictions = (probabilities > best_threshold).astype(int)
   ```

**B. Experiment Tracking**

```python
class ExperimentTracker:
    """Logs model experiments and results (MLflow-style)"""

    def log_parameters(self, params):
        """Log hyperparameters"""
        # learning_rate, n_estimators, max_depth, etc.

    def log_metrics(self, metrics):
        """Log evaluation metrics"""
        # precision, recall, f1, roc_auc, etc.

    def log_model(self, model, model_name):
        """Save trained model with metadata"""

    def compare_models(self, experiment_ids):
        """Compare multiple experiment results"""
```

**What to Track**:
- Hyperparameters (learning rate, tree depth, etc.)
- Metrics (precision, recall, ROC-AUC)
- Training time, data version, code version
- Feature importance
- Confusion matrix, PR curve, ROC curve

**C. Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.3]
}

# Use TimeSeriesSplit for CV
grid_search = GridSearchCV(
    estimator=XGBClassifier(),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='recall',  # Optimize for recall
    n_jobs=-1
)
```

**Optimization Methods**:
- **Grid Search**: Exhaustive, best for small param space
- **Random Search**: Sample randomly, faster
- **Bayesian Optimization**: Smart search (Optuna, Hyperopt)
- **Early Stopping**: Stop if no improvement

**D. Model Versioning**

```python
class ModelRegistry:
    """Manages model versions and deployment"""

    def register_model(self, model, version, metrics, metadata):
        """Save model with version tag"""

    def promote_to_production(self, model_version):
        """Mark model as production-ready"""

    def rollback_to_version(self, version):
        """Revert to previous model"""

    def compare_versions(self, v1, v2):
        """Compare model performance"""
```

#### Interview Questions to Prepare

**Q: How do you prevent overfitting?**
- Cross-validation with proper time-based splits
- Regularization (L1/L2 penalties)
- Early stopping based on validation performance
- Feature selection (remove correlated features)
- Ensemble methods (reduce variance)

**Q: How do you choose between models?**
- Primary metric: Recall (catch at-risk customers)
- Secondary: Precision (avoid false alarms)
- Business metric: ROI of retention campaign
- Latency: Can it serve predictions fast enough?
- Interpretability: Can we explain predictions?
- Stability: Consistent performance over time?

**Q: What's your retraining strategy?**
- **Scheduled**: Weekly/monthly retraining
- **Triggered**: When performance drops >5%
- **Continuous**: Online learning for some models
- **Trade-off**: Freshness vs stability vs cost

---

### 4. Model Evaluation

#### Purpose
Assess model performance using multiple metrics and business criteria.

#### Evaluation Metrics

**A. Classification Metrics**

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

class ModelEvaluator:
    """Comprehensive model evaluation"""

    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Compute all relevant metrics"""
        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'pr_auc': average_precision_score(y_true, y_proba)
        }
```

**Why Different Metrics Matter**:

| Metric | What it Measures | When to Use |
|--------|-----------------|-------------|
| **Precision** | Of predicted churners, % actually churn | Minimize wasted campaigns |
| **Recall** | Of actual churners, % we catch | Minimize missed opportunities |
| **F1-Score** | Harmonic mean of precision & recall | Balanced performance |
| **ROC-AUC** | Overall discrimination ability | Compare models |
| **PR-AUC** | Performance on imbalanced data | Better than ROC for imbalance |

**B. Confusion Matrix Analysis**

```
                  Predicted
                  No Churn  |  Churn
Actual   ───────────────────────────────
No Churn │   TN (85,000) │ FP (1,000)
Churn    │   FN (3,000)  │ TP (11,000)
```

**Business Impact**:
- **True Positive (TP)**: Correctly identified churners → Saved customers
- **False Positive (FP)**: Incorrect churn prediction → Wasted campaign
- **False Negative (FN)**: Missed churners → Lost revenue
- **True Negative (TN)**: Correctly identified retained → No action needed

**C. Threshold Tuning for Business Objectives**

```python
def tune_threshold(y_true, y_proba, campaign_cost=10, customer_value=600, intervention_success=0.3):
    """Find threshold that maximizes ROI"""

    thresholds = np.linspace(0.1, 0.9, 100)
    best_roi = -float('inf')
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # ROI calculation
        saved_revenue = tp * customer_value * intervention_success
        wasted_cost = fp * campaign_cost
        missed_revenue = fn * customer_value

        roi = saved_revenue - wasted_cost

        if roi > best_roi:
            best_roi = roi
            best_threshold = threshold

    return best_threshold, best_roi
```

**D. Feature Importance Analysis**

```python
def analyze_feature_importance(model, feature_names):
    """Understand which features drive predictions"""

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Top 10 Most Important Features:")
    for i in range(10):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```

**Interview Tip**: Discuss which features are most predictive and why.

#### Interview Questions to Prepare

**Q: Why is accuracy a poor metric for imbalanced data?**
- With 85% non-churn, a model that always predicts "no churn" has 85% accuracy but 0% recall!
- Accuracy treats all errors equally, but FN (missing churners) is more costly than FP.

**Q: How do you choose the decision threshold?**
1. Default 0.5 is arbitrary
2. Analyze precision-recall trade-off
3. Calculate business ROI at different thresholds
4. Consider capacity constraints (can only contact 1,000 customers/day)
5. Use different thresholds for different segments (high-value customers: lower threshold)

**Q: What if model performance degrades over time?**
- Monitor metrics on recent data
- Check for data drift (feature distributions changed)
- Check for concept drift (relationship between X and y changed)
- Retrain with recent data
- Investigate root cause (seasonality, external events, data quality issues)

---

### 5. Model Serving (Inference Pipeline)

#### Purpose
Deploy model to production for real-time and batch predictions.

#### Components

**A. Real-time Inference API**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model at startup
model = joblib.load('models/churn_model_v1.pkl')
feature_pipeline = joblib.load('models/feature_pipeline_v1.pkl')

class PredictionRequest(BaseModel):
    customer_id: str
    demographics: dict
    usage_metrics: dict

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: str
    risk_level: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """Real-time churn prediction endpoint"""

    try:
        # 1. Feature engineering
        features = feature_pipeline.transform(request)

        # 2. Model prediction
        probability = model.predict_proba(features)[0][1]
        prediction = "churn" if probability > 0.35 else "no_churn"

        # 3. Risk categorization
        if probability > 0.7:
            risk = "high"
        elif probability > 0.4:
            risk = "medium"
        else:
            risk = "low"

        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=probability,
            prediction=prediction,
            risk_level=risk
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**B. Batch Prediction**

```python
class BatchPredictor:
    """Score entire customer base daily"""

    def score_all_customers(self, customer_data):
        """Batch prediction for 1M customers"""

        # Process in chunks to avoid memory issues
        chunk_size = 10000
        predictions = []

        for i in range(0, len(customer_data), chunk_size):
            chunk = customer_data[i:i+chunk_size]
            features = self.feature_pipeline.transform(chunk)
            proba = self.model.predict_proba(features)[:, 1]
            predictions.extend(proba)

        return predictions

    def save_predictions(self, customer_ids, probabilities, date):
        """Store predictions for downstream use"""
        # Save to database or data lake
```

**C. Model Caching & Optimization**

```python
class PredictionService:
    """Optimized inference service"""

    def __init__(self):
        self.model = self.load_model()
        self.feature_cache = LRUCache(maxsize=10000)

    def predict(self, customer_id, features):
        """Predict with caching"""

        # Check cache first
        cache_key = f"{customer_id}:{hash(features)}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Compute prediction
        prediction = self.model.predict_proba([features])[0][1]

        # Cache result (TTL: 1 hour)
        self.feature_cache[cache_key] = prediction

        return prediction
```

**D. A/B Testing Support**

```python
class ABTestingRouter:
    """Route traffic between model versions"""

    def route_prediction(self, customer_id, features):
        """Route to control or treatment model"""

        # Hash customer_id to assign to group
        group = hash(customer_id) % 100

        if group < 10:  # 10% to new model
            return self.model_v2.predict(features)
        else:  # 90% to current model
            return self.model_v1.predict(features)
```

#### Interview Questions to Prepare

**Q: How do you achieve <100ms latency for predictions?**
- **Model optimization**: Use simpler model or model compression
- **Feature caching**: Cache computed features
- **Result caching**: Cache predictions (TTL depends on feature freshness)
- **Batch predictions**: Precompute for known queries
- **Hardware**: Use GPUs for large models, optimize CPU usage
- **Load balancing**: Distribute load across multiple servers

**Q: How do you handle model updates without downtime?**
- **Blue-green deployment**: Switch traffic to new model after validation
- **Canary deployment**: Route small % of traffic to new model first
- **Shadow mode**: Run new model in parallel, don't affect production
- **Feature flags**: Toggle models dynamically
- **Rollback plan**: Keep old model available for quick revert

**Q: What if the model fails during inference?**
- **Graceful degradation**: Return conservative prediction (e.g., low churn risk)
- **Circuit breaker**: Stop calling failing service, retry after cooldown
- **Fallback model**: Use simpler rule-based model
- **Alerting**: Notify on-call engineer immediately
- **Logging**: Capture error details for debugging

---

### 6. Monitoring & Observability

#### Purpose
Detect issues early and trigger retraining when needed.

#### What to Monitor

**A. Model Performance Metrics**

```python
class PerformanceMonitor:
    """Track model performance over time"""

    def track_daily_metrics(self, y_true, y_pred, date):
        """Log daily performance metrics"""
        metrics = {
            'date': date,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

        # Alert if metrics drop >5%
        if metrics['recall'] < self.baseline_recall * 0.95:
            self.send_alert("Recall dropped below threshold")
```

**B. Data Drift Detection**

**Problem**: Feature distributions change over time (new user demographics, usage patterns shift).

```python
from scipy.stats import ks_2samp

class DriftDetector:
    """Detect distribution changes in features"""

    def detect_drift(self, training_data, production_data, features):
        """Compare feature distributions using KS test"""

        drift_detected = {}

        for feature in features:
            train_dist = training_data[feature]
            prod_dist = production_data[feature]

            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(train_dist, prod_dist)

            if p_value < 0.05:  # Significant difference
                drift_detected[feature] = {
                    'statistic': statistic,
                    'p_value': p_value
                }

        return drift_detected
```

**Drift Detection Methods**:
- **KS Test**: Statistical test for distribution difference
- **PSI (Population Stability Index)**: Measures distribution shift
- **Jensen-Shannon Divergence**: Symmetric measure of distribution difference
- **Visual**: Plot distributions over time

**C. Prediction Distribution Monitoring**

```python
def monitor_prediction_distribution(predictions, date):
    """Ensure prediction distribution is stable"""

    churn_rate = np.mean(predictions > 0.35)

    # Alert if churn rate suddenly changes
    if churn_rate > 0.25 or churn_rate < 0.10:
        send_alert(f"Unusual churn rate on {date}: {churn_rate:.2%}")
```

**D. Latency & Throughput Monitoring**

```python
import time
from prometheus_client import Histogram, Counter

# Metrics
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
prediction_count = Counter('predictions_total', 'Total predictions')

def predict_with_monitoring(customer_id, features):
    """Predict with latency tracking"""

    start_time = time.time()

    prediction = model.predict(features)

    latency = time.time() - start_time
    prediction_latency.observe(latency)
    prediction_count.inc()

    if latency > 0.1:  # >100ms
        log_slow_prediction(customer_id, latency)

    return prediction
```

**E. Retraining Triggers**

```python
class RetrainingTrigger:
    """Decide when to retrain model"""

    def should_retrain(self, current_metrics, drift_status, days_since_training):
        """Determine if retraining is needed"""

        # Trigger 1: Performance degradation
        if current_metrics['recall'] < self.baseline_recall * 0.95:
            return True, "Performance degraded >5%"

        # Trigger 2: Significant drift
        if len(drift_status) > 5:
            return True, f"Drift detected in {len(drift_status)} features"

        # Trigger 3: Scheduled retraining
        if days_since_training > 30:
            return True, "Scheduled monthly retraining"

        return False, "No retraining needed"
```

#### Interview Questions to Prepare

**Q: What's the difference between data drift and concept drift?**
- **Data Drift (Covariate Shift)**: Distribution of input features changes (P(X) changes)
  - Example: User demographics shift (younger users signing up)
  - Detection: Compare feature distributions
  - Action: Retrain with recent data

- **Concept Drift**: Relationship between X and y changes (P(y|X) changes)
  - Example: What causes churn changes (new competitor, price increase)
  - Detection: Monitor model performance on labeled data
  - Action: Retrain and possibly reengineer features

**Q: How do you monitor a model without immediate ground truth?**
- **Proxy metrics**: User engagement after prediction (did they click retention offer?)
- **Delayed feedback**: Wait 30 days to see who actually churned
- **Drift detection**: Monitor input distribution changes
- **Business metrics**: Track retention rate, campaign success rate
- **A/B testing**: Compare model versions on business outcomes

**Q: What metrics would you put on a dashboard?**
- **Model Performance**: Precision, recall, F1 (daily, rolling 7-day avg)
- **Predictions**: Churn rate predicted, high-risk customer count
- **Drift**: Features with significant drift, magnitude
- **System Health**: API latency (p50, p95, p99), error rate, throughput
- **Business Impact**: Customers saved, ROI, retention rate

---

## Technology Stack

### Core ML Libraries
- **scikit-learn**: Logistic Regression, Random Forest, preprocessing
- **XGBoost**: Gradient boosting for better performance
- **imbalanced-learn**: SMOTE for handling class imbalance
- **pandas, numpy**: Data manipulation

### Experimentation & Tracking
- **MLflow**: Experiment tracking, model registry
- **Optuna**: Hyperparameter optimization (alternative: Hyperopt)

### Feature Store
- **Feast**: Open-source feature store
- **Alternative**: Custom implementation with SQLite/Postgres

### Model Serving
- **FastAPI**: REST API framework (async, high performance)
- **Uvicorn**: ASGI server
- **Pydantic**: Request/response validation

### Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Evidently**: Data drift detection
- **Python logging**: Application logs

### Data Processing
- **SQLAlchemy**: Database ORM
- **Apache Spark**: Large-scale batch processing (if needed)
- **Kafka**: Streaming data ingestion (if needed)

### Testing
- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **locust**: Load testing for API

---

## Implementation Priorities

### Phase 1: Core Pipeline (MVP)
1. Data generation and ingestion
2. Feature engineering
3. Model training (1-2 models)
4. Basic evaluation
5. Simple inference (batch predictions)

### Phase 2: Production Readiness
6. REST API for real-time predictions
7. Experiment tracking
8. Model versioning
9. Comprehensive testing

### Phase 3: Monitoring & Operations
10. Drift detection
11. Performance monitoring
12. Alerting system
13. Retraining pipeline

---

## Key Design Decisions & Trade-offs

### 1. Model Complexity vs Interpretability
- **Complex (XGBoost)**: Higher accuracy, harder to explain
- **Simple (Logistic Regression)**: Lower accuracy, easy to explain
- **Decision**: Use XGBoost for predictions, Logistic Regression for feature importance analysis

### 2. Real-time vs Batch Predictions
- **Real-time**: Low latency (<100ms), higher infrastructure cost
- **Batch**: Process all customers overnight, lower cost, stale predictions
- **Decision**: Hybrid approach (batch daily, real-time for critical paths like checkout)

### 3. Feature Store: Build vs Buy
- **Build**: Custom solution, lower cost, limited features
- **Buy**: Feast/Tecton, more features, learning curve
- **Decision**: Start with simple SQLite, migrate to Feast if scaling issues

### 4. Retraining Frequency
- **Daily**: Fresh model, high cost, risk of instability
- **Monthly**: Lower cost, model drift risk
- **Decision**: Weekly retraining + trigger-based (performance drop)

See [trade_offs.md](trade_offs.md) for detailed analysis.

---

## Success Metrics

### Technical Metrics
- Recall >0.75 (catch 75% of churners)
- Precision >0.60 (60% of predicted churners actually churn)
- API latency <100ms (p99)
- System uptime >99.9%

### Business Metrics
- Reduce churn rate by 2 percentage points
- Achieve $500K+ in retained revenue per year
- Retention campaign ROI >300%

---

## Next Steps

1. **Study Implementation**: Review source code in `src/` directory
2. **Run Notebooks**: Execute end-to-end examples
3. **Practice Explanations**: Articulate design decisions
4. **Review Interview Questions**: Prepare for common questions
5. **Experiment**: Modify features, try different models

This solution approach provides a comprehensive blueprint for building a production ML pipeline. Focus on understanding the **why** behind each design decision for interview success.
