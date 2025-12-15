# Solution Approach: Credit Card Fraud Detection

## Overview

This solution demonstrates how to handle severely imbalanced classification problems using fraud detection as a real-world example. The focus is on understanding core concepts rather than building a production system.

## Architecture

```
Raw Transaction Data
        ↓
[Data Preprocessing]
    - Handle missing values
    - Feature scaling
    - Train/test split
        ↓
[Class Imbalance Handling]
    - SMOTE oversampling
    - Class weights
        ↓
[Model Training]
    - Random Forest (baseline)
    - XGBoost (advanced)
        ↓
[Threshold Tuning]
    - Optimize for business metrics
    - Precision-recall trade-off
        ↓
[Evaluation]
    - F1 Score, PR-AUC, ROC-AUC
    - Confusion matrix analysis
        ↓
[Inference]
    - Predict fraud probability
    - Apply threshold
```

## Key Design Decisions

### 1. Model Selection: Tree-Based Models

**Why Random Forest and XGBoost?**

**Advantages for Fraud Detection:**
- **Handle imbalanced data well** with class weights
- **Capture non-linear patterns** (fraud patterns are complex)
- **Feature importance** (explain why transaction was flagged - regulatory requirement)
- **Robust to outliers** (unusual transactions are common)
- **No feature scaling required** (unlike neural networks, SVM)
- **Fast inference** (decision trees are quick to evaluate)

**Interview Tip**: Mention that deep learning might achieve slightly higher accuracy, but tree-based models offer better interpretability, which is critical in financial applications for regulatory compliance.

### 2. Handling Class Imbalance

We implement **three complementary techniques**:

#### A. SMOTE (Synthetic Minority Over-sampling Technique)

```python
from imblearn.over_sampling import SMOTE

# Creates synthetic fraud examples by interpolating between existing fraud cases
smote = SMOTE(sampling_strategy=0.3)  # Increase fraud to 30% of legitimate
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**How it works**:
1. For each fraud transaction, find k nearest fraud neighbors
2. Create synthetic samples by interpolating features
3. Balance the dataset without just duplicating examples

**Trade-off**: May create unrealistic samples. Only apply to training data, never test data.

#### B. Class Weights

```python
# Automatically adjusts loss function to penalize misclassifying fraud more
model = RandomForestClassifier(class_weight='balanced')
```

**How it works**:
- Gives higher weight to minority class errors
- Fraud misclassification costs more in the loss function
- Model learns to prioritize detecting fraud

**Interview Insight**: This is simpler than SMOTE and often works just as well. Explain that you'd try this first in a real project.

#### C. Custom Threshold Tuning

```python
# Instead of default 0.5 threshold, optimize for business metric
# Lower threshold = catch more fraud but more false positives
optimal_threshold = 0.3  # Determined by precision-recall curve
```

**Why it matters**: Default 0.5 threshold assumes equal costs for FP and FN. In fraud, catching fraud is more valuable than avoiding false alarms.

### 3. Evaluation Metrics

**Why NOT Accuracy?**

With 98% legitimate transactions, a dummy model predicting "all legitimate" achieves 98% accuracy but 0% fraud detection.

**What We Use Instead:**

#### Precision-Recall Metrics

```python
# Precision: When we predict fraud, how often are we right?
precision = TP / (TP + FP)

# Recall: Of all actual fraud, how much did we catch?
recall = TP / (TP + FN)

# F1: Harmonic mean of precision and recall
f1 = 2 * (precision * recall) / (precision + recall)
```

**Interview Explanation**:
- **High Recall** (e.g., 85%) = "We catch 85% of fraud"
- **Moderate Precision** (e.g., 25%) = "When we flag fraud, we're right 25% of the time"

In fraud detection, **recall is more important** (catching fraud) but we need **acceptable precision** (don't overwhelm investigators with false alarms).

#### PR-AUC (Precision-Recall Area Under Curve)

- Better than ROC-AUC for imbalanced data
- Shows model performance across all thresholds
- Higher is better (1.0 = perfect)

#### Confusion Matrix

```
                Predicted
                Leg    Fraud
Actual  Leg     TN     FP
        Fraud   FN     TP
```

**What matters**:
- **TP (True Positive)**: Correctly caught fraud ✓
- **FN (False Negative)**: Missed fraud (costly!) ✗✗
- **FP (False Positive)**: False alarm (annoying but acceptable)
- **TN (True Negative)**: Normal operation ✓

### 4. Feature Engineering

For this learning example, we use simplified features:

```python
Features:
- Time: Seconds since first transaction (detects unusual timing)
- Amount: Transaction dollar value (high amounts = higher risk)
- V1-V28: PCA-transformed anonymized features
```

**In Production, You'd Add**:
- Transaction velocity (transactions per hour)
- Geographic features (distance from last transaction)
- Merchant category codes
- Device/IP fingerprinting
- Historical user behavior patterns

**Interview Tip**: Mention that feature engineering is often more important than model choice. Simple models with great features beat complex models with raw features.

### 5. Implementation Details

#### Data Preprocessing

```python
# 1. Handle missing values (if any)
# 2. Split data BEFORE any resampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 3. Scale features (for some models, not tree-based)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training
X_test_scaled = scaler.transform(X_test)        # Apply same transform to test
```

**Critical Mistake to Avoid**: Never resample before splitting! This causes data leakage.

#### Model Training

```python
# Baseline: Random Forest with class weights
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',  # Handle imbalance
    random_state=42
)

# Advanced: XGBoost with scale_pos_weight
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    scale_pos_weight=99,  # Ratio of negative to positive (99:1)
    random_state=42
)
```

#### Threshold Tuning

```python
# Get probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate precision-recall for different thresholds
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Choose threshold based on business requirements
# Example: Want recall >= 80%
optimal_idx = np.where(recalls >= 0.80)[0][0]
optimal_threshold = thresholds[optimal_idx]
```

## Code Structure

### Module Organization

```
src/
├── data_preprocessor.py    # Load data, clean, split, scale
├── fraud_detector.py        # Model training, prediction, evaluation
└── utils.py                 # Helper functions, metrics
```

**Why Simple Structure**: For learning, we want clear, easy-to-follow modules. Each module has a single responsibility.

### Class Design

```python
class FraudDetector:
    """
    Simple fraud detection model with clear methods for learning.
    """

    def train(self, X_train, y_train, use_smote=True):
        """Train model with optional SMOTE oversampling."""

    def predict(self, X, threshold=0.5):
        """Predict fraud with custom threshold."""

    def evaluate(self, X_test, y_test):
        """Calculate all relevant metrics."""

    def plot_metrics(self, X_test, y_test):
        """Visualize precision-recall curve, confusion matrix."""
```

## Expected Results

With our synthetic dataset, you should see:

**Before Class Imbalance Handling**:
- Accuracy: ~98% (misleading!)
- Recall: ~40% (missing 60% of fraud)
- Precision: ~15%

**After SMOTE + Class Weights**:
- Accuracy: ~95% (lower but that's okay)
- Recall: ~80% (catching 80% of fraud)
- Precision: ~25% (acceptable false positive rate)
- F1 Score: ~0.38
- PR-AUC: ~0.65

**Interview Insight**: Notice accuracy went DOWN but fraud detection went UP. This is the right trade-off!

## Common Interview Questions & Answers

### Q: Why not use accuracy?

**A**: "With 98% legitimate transactions, a model that predicts 'legitimate' for everything achieves 98% accuracy but catches zero fraud. For imbalanced data, we need metrics that focus on the minority class, like precision, recall, and F1 score."

### Q: SMOTE vs Class Weights - which is better?

**A**: "I'd start with class weights because it's simpler and doesn't modify the data. If that doesn't work well, I'd try SMOTE. In practice, combining both often gives the best results. The key is to validate on unmodified test data."

### Q: How would you deploy this in production?

**A**: "I'd consider:
1. **Real-time inference**: Sub-100ms prediction latency
2. **Feature serving**: Pre-compute features where possible
3. **Model serving**: REST API or embedded model
4. **Monitoring**: Track precision/recall over time to detect drift
5. **A/B testing**: Gradually roll out new models
6. **Feedback loop**: Investigate flagged transactions to retrain model
7. **Fallback**: Rule-based system if model fails"

### Q: What if fraud patterns change over time?

**A**: "This is called concept drift. I'd:
1. **Monitor metrics**: Track precision/recall daily
2. **Retrain regularly**: Weekly or monthly retraining
3. **Online learning**: Update model with new fraud cases
4. **Ensemble**: Combine recent and historical models
5. **Alert system**: Flag when metrics degrade"

## Limitations & Future Improvements

**Current Limitations**:
- Small dataset (1,000 transactions)
- Static model (no retraining)
- Single model (no ensemble)
- Simplified features
- No real-time inference

**Production Enhancements**:
- Ensemble of models (RF + XGBoost + Neural Network)
- Deep learning for complex patterns
- Real-time feature engineering
- Anomaly detection for novel fraud
- Explainable AI (SHAP values for regulatory compliance)

## Key Takeaways for Interviews

1. **Class imbalance is the core challenge** - understand why and how to handle it
2. **Metrics matter more than model choice** - precision, recall, F1 for imbalanced data
3. **Business context drives decisions** - threshold tuning based on cost of FP vs FN
4. **Interpretability is crucial** - tree-based models for explainability
5. **Evaluation must be rigorous** - never resample test data, stratified splits
