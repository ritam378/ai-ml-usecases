# Customer Churn Dataset - Data Description

## Overview

This dataset contains **10,000 synthetic customer records** for a subscription-based streaming service (Netflix/Spotify-style). It's designed to simulate realistic customer behavior patterns for churn prediction modeling.

- **Total Records**: 10,000 customers
- **Features**: 32 (including derived features)
- **Target Variable**: `churned` (binary: 0 = retained, 1 = churned)
- **Churn Rate**: ~15% (realistic class imbalance)
- **Time Period**: 2-year customer history (2023-12-16 to 2025-12-15)
- **File Formats**: CSV (1.5 MB), JSON (9.3 MB)

---

## Schema Reference

### Identifier
| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `customer_id` | string | Unique customer identifier | `CUST_000001` |

### Demographics (4 features)
| Feature | Type | Range | Description | Churn Correlation |
|---------|------|-------|-------------|-------------------|
| `age` | int | 18-75 | Customer age (normal distribution, mean=40) | Weak |
| `gender` | string | M/F | Customer gender (50/50 split) | None |
| `location` | string | 10 cities | Customer location (US major cities) | Weak |
| `data_date` | date | - | Snapshot date (2025-12-15) | - |

**Distribution**:
- Age: Normal distribution (μ=40, σ=15)
- Cities: New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, San Jose

---

### Subscription Information (5 features)
| Feature | Type | Range | Description | Churn Correlation |
|---------|------|-------|-------------|-------------------|
| `signup_date` | datetime | 30-730 days ago | When customer signed up | **Strong** (new customers churn more) |
| `tenure_days` | int | 30-730 | Days since signup | **Strong** (negative correlation) |
| `plan_type` | string | Basic/Standard/Premium | Subscription tier | Moderate (Basic churns more) |
| `monthly_price` | float | $9.99-$19.99 | Monthly subscription cost | Moderate |
| `payment_method` | string | 4 options | How customer pays | Weak |

**Key Insights**:
- **Tenure**: Customers with <180 days tenure have **2-3x higher churn rate**
- **Plan Distribution**:
  - New customers (<1 year): 50% Basic, 30% Standard, 20% Premium
  - Old customers (>1 year): 20% Basic, 30% Standard, 50% Premium
- **Payment Methods**: Credit Card (50%), Debit Card (25%), PayPal (15%), Bank Transfer (10%)

---

### Usage Patterns (7 features)
| Feature | Type | Range | Description | Churn Correlation |
|---------|------|-------|-------------|-------------------|
| `logins_last_7d` | int | 0-10 | Login count in last 7 days | **Very Strong** |
| `logins_last_30d` | int | 0-40 | Login count in last 30 days | **Very Strong** |
| `logins_last_90d` | int | 0-120 | Login count in last 90 days | **Strong** |
| `avg_session_duration` | float | 5-60 min | Average session length (minutes) | Strong |
| `total_watch_time_30d` | float | 0-100 hrs | Total watch time in last 30 days | Strong |
| `unique_content_30d` | int | 0-50 | Unique items watched (last 30 days) | Moderate |
| `days_since_last_login` | int | 0-30 | Days since last login | **Very Strong** |

**Key Insights**:
- **Churners** have **70% lower usage** across all metrics
- `logins_last_30d` < 5 → **High churn risk** (80%+ churn rate)
- `days_since_last_login` > 14 → **Very high risk** (90%+ churn rate)
- Normal distribution for active users, but churners cluster at low values

**Interview Tip**: These are **leading indicators** of churn. Discuss how to use them for early intervention (e.g., send re-engagement email when logins drop below threshold).

---

### Engagement Features (6 features)
| Feature | Type | Range | Description | Churn Correlation |
|---------|------|-------|-------------|-------------------|
| `support_tickets_30d` | int | 0-5 | Support ticket count (last 30 days) | Moderate (U-shaped) |
| `satisfaction_score` | float | 1.0-5.0 | Customer satisfaction rating | **Strong** (negative) |
| `app_rating` | float | 1.0-5.0 | App store rating submitted | Moderate |
| `social_connections` | int | 0-50 | Number of friends/followers | Weak |
| `feature_adoption_rate` | float | 0.0-1.0 | % of platform features used | Strong |
| `data_date` | date | - | Date of data snapshot | - |

**Key Insights**:
- **Satisfaction Score**:
  - Churners: Mean = 2.5 (σ=0.5)
  - Active: Mean = 4.5 (σ=0.3)
  - Score < 3.0 → **High churn risk**
- **Support Tickets**: U-shaped relationship
  - 0 tickets: Low engagement (moderate churn)
  - 1-2 tickets: Engaged customers seeking help (low churn)
  - 3+ tickets: Frustrated customers (high churn)
- **Feature Adoption**: Correlated with engagement and retention

---

### Transaction Features (5 features)
| Feature | Type | Range | Description | Churn Correlation |
|---------|------|-------|-------------|-------------------|
| `payment_failures_6m` | int | 0-5 | Payment failures in last 6 months | **Very Strong** |
| `plan_changes_12m` | int | 0-3 | Plan upgrades/downgrades (12 months) | Weak |
| `billing_disputes` | int | 0-2 | Billing disputes raised | Strong |
| `auto_renew_enabled` | int | 0/1 | Auto-renewal enabled (boolean) | **Strong** |
| `days_until_renewal` | int | 1-30 | Days until next billing cycle | Weak |

**Key Insights**:
- **Payment Failures**: **Strongest predictor** of churn
  - Churners: 80% have 2+ payment failures
  - Active: 90% have 0 payment failures
  - Interview tip: This is often due to expired cards, not intentional churn
- **Auto-Renewal**:
  - Enabled → 10% churn rate
  - Disabled → 50% churn rate (5x higher!)
- **Billing Disputes**: Strong signal but rare (only 5% of customers)

---

### Derived Features (4 features)
| Feature | Type | Range | Description | How It's Calculated |
|---------|------|-------|-------------|---------------------|
| `login_frequency_trend` | float | 0.0-1.0 | Login trend (recent vs historical) | `logins_last_7d / (logins_last_30d / 4.3)` |
| `engagement_score` | float | 0.0-1.0 | Composite engagement metric | Weighted avg of satisfaction, logins, watch time, feature adoption |
| `high_support_flag` | int | 0/1 | Frustrated customer indicator | 1 if `support_tickets_30d >= 3` |
| `payment_issue_flag` | int | 0/1 | Payment problem indicator | 1 if `payment_failures_6m >= 2` |
| `low_engagement_flag` | int | 0/1 | Low engagement indicator | 1 if `logins_last_30d < 5` OR `days_since_last_login > 14` |

**Key Insights**:
- **Engagement Score**: Weighted composite feature
  - Formula: `0.3*satisfaction + 0.3*login_norm + 0.2*watch_time_norm + 0.2*feature_adoption`
  - Churners: Mean = 0.25
  - Active: Mean = 0.55
- **Login Frequency Trend**:
  - Value < 0.5 → Usage is declining (churn risk)
  - Value > 1.0 → Usage is increasing (retention)
- **Risk Flags**: Binary indicators for quick filtering
  - Customers with 2+ flags → 80% churn rate
  - Customers with 0 flags → 5% churn rate

**Interview Tip**: Discuss why derived features improve model performance (they encode domain knowledge and capture non-linear relationships).

---

## Target Variable

### `churned` (Binary Classification)

| Value | Label | Count | Percentage | Description |
|-------|-------|-------|------------|-------------|
| 0 | Retained | 8,500 | 85% | Customer is still active |
| 1 | Churned | 1,500 | 15% | Customer canceled subscription |

**Churn Logic** (Probabilistic):

Churn probability is calculated based on weighted features:
```python
churn_prob = (
    0.35 * (payment_failures >= 2) +           # Payment issues
    0.25 * (logins_last_30d < 5) +             # Low activity
    0.20 * (satisfaction_score < 3.0) +        # Dissatisfaction
    0.10 * (auto_renew_enabled == 0) +         # No auto-renewal
    0.10 * (days_since_last_login > 14)        # Recent inactivity
)
```

**Class Imbalance**:
- This is a **realistic imbalance** (typical churn rates: 10-20%)
- **Interview discussion points**:
  - Why accuracy is a poor metric (85% accuracy by predicting all "no churn")
  - Need for class_weight='balanced', SMOTE, or threshold tuning
  - Business cost asymmetry (false negative = lose $600 CLV, false positive = $10 incentive cost)

---

## Feature Importance (Expected)

Based on the data generation logic, expected feature importance:

| Rank | Feature | Type | Why It's Important |
|------|---------|------|-------------------|
| 1 | `payment_failures_6m` | Transaction | **Strongest signal**: 2+ failures → 80% churn |
| 2 | `logins_last_30d` | Usage | **Leading indicator**: <5 logins → high risk |
| 3 | `days_since_last_login` | Usage | **Recency matters**: >14 days → very high risk |
| 4 | `satisfaction_score` | Engagement | **Direct feedback**: <3.0 → dissatisfied |
| 5 | `auto_renew_enabled` | Transaction | **Intent signal**: disabled → 5x higher churn |
| 6 | `engagement_score` | Derived | **Composite metric**: captures overall behavior |
| 7 | `tenure_days` | Subscription | **Loyalty proxy**: newer customers churn more |
| 8 | `login_frequency_trend` | Derived | **Trend detection**: declining usage → risk |
| 9 | `low_engagement_flag` | Derived | **Binary risk indicator**: simple rule-based |
| 10 | `payment_issue_flag` | Derived | **Payment health check**: actionable signal |

**Interview Tip**: Discuss how you'd validate feature importance using:
- Permutation importance (model-agnostic)
- SHAP values (explains individual predictions)
- Business validation (correlate with actual churn reasons from surveys)

---

## Data Quality & Characteristics

### Completeness
- **No missing values**: All fields are populated
- **Interview discussion**: In production, discuss how to handle missing data (imputation strategies, missingness as a feature)

### Realistic Correlations
The data includes realistic correlations:
1. **Tenure → Plan Type**: Older customers have premium plans
2. **Churn → Low Usage**: Churners have 70% lower usage
3. **Payment Failures → Churn**: 80% of churners have payment issues
4. **Satisfaction → Engagement**: High satisfaction = high engagement
5. **Auto-Renewal OFF → Higher Churn**: 5x churn rate

### Temporal Considerations
- **Signup Date Range**: 30 to 730 days ago (2-year history)
- **Time-based Split Required**: Use `signup_date` or `tenure_days` for proper train/test split
- **Avoid Data Leakage**: Don't use future information (e.g., `days_until_renewal` for predictions made >30 days in advance)

**Interview Tip**: Discuss **point-in-time correctness**:
- Training data: Use features as they existed at time of churn decision
- Serving: Ensure same feature calculation logic (avoid training/serving skew)

---

## Usage Guidelines for ML Pipeline

### 1. Loading the Data
```python
import pandas as pd

# Load CSV
df = pd.read_csv('data/customers.csv')

# Parse dates
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['data_date'] = pd.to_datetime(df['data_date'])

print(f"Shape: {df.shape}")
print(f"Churn rate: {df['churned'].mean():.1%}")
```

### 2. Feature Selection
**Recommended feature groups**:

```python
# Baseline features (start simple)
baseline_features = [
    'tenure_days', 'logins_last_30d', 'satisfaction_score',
    'payment_failures_6m', 'auto_renew_enabled'
]

# Advanced features (add complexity)
advanced_features = baseline_features + [
    'logins_last_7d', 'logins_last_90d', 'avg_session_duration',
    'total_watch_time_30d', 'days_since_last_login',
    'feature_adoption_rate', 'billing_disputes'
]

# With derived features (best performance)
full_features = advanced_features + [
    'engagement_score', 'login_frequency_trend',
    'high_support_flag', 'payment_issue_flag', 'low_engagement_flag'
]
```

### 3. Train/Test Split
```python
from sklearn.model_selection import train_test_split

# Option 1: Random split (not recommended for time-series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Option 2: Temporal split (RECOMMENDED)
# Train on customers who signed up before a date, test on recent
cutoff_date = pd.Timestamp('2025-09-01')
train_mask = df['signup_date'] < cutoff_date
test_mask = df['signup_date'] >= cutoff_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
```

**Interview Tip**: Explain why temporal split is more realistic (simulates production scenario where you predict future churn based on past data).

### 4. Handling Class Imbalance
```python
# Method 1: Class weights (simplest)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

# Method 2: SMOTE (synthetic oversampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Method 3: Threshold tuning (most important!)
y_proba = model.predict_proba(X_test)[:, 1]
optimal_threshold = 0.3  # Find using business cost analysis
y_pred = (y_proba >= optimal_threshold).astype(int)
```

### 5. Feature Preprocessing
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Numeric features: scale
numeric_features = ['age', 'tenure_days', 'logins_last_30d', ...]
numeric_transformer = StandardScaler()

# Categorical features: encode
categorical_features = ['gender', 'location', 'plan_type', 'payment_method']
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(scale_pos_weight=5.67))  # 8500/1500
])
```

### 6. Model Evaluation
```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Business metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
cost_fp = 10   # Cost of retention offer
cost_fn = 600  # CLV lost from false negative

total_cost = (fp * cost_fp) + (fn * cost_fn)
print(f"Total cost: ${total_cost:,}")
```

**Interview Tip**: Always connect technical metrics to business outcomes. ROC-AUC is great, but executives care about "How much money did we save?"

---

## Common Interview Questions About This Dataset

### Q1: Why is the churn rate 15%? Why not 50% (balanced)?
**Answer**: Real-world churn rates are typically 10-20% for subscription businesses. Using realistic imbalance teaches you to handle class imbalance, which is critical for production ML systems. A 50/50 split would be unrealistic and wouldn't prepare you for real challenges.

### Q2: Why not include more features (100+ features)?
**Answer**: This dataset focuses on **quality over quantity**. The 32 features represent the most important categories (usage, engagement, transactions) that would exist in a real churn system. Adding 100+ features would make it harder to learn feature engineering principles and explain model decisions.

### Q3: How do you prevent data leakage with this dataset?
**Answer**: Potential leakage sources:
1. **Temporal leakage**: Don't use `days_until_renewal` for predictions made >30 days out
2. **Target leakage**: `churn_probability_true` is the "true" probability used to generate labels—**never use this as a feature!**
3. **Future information**: In production, ensure all features are calculated using only data available at prediction time

### Q4: What if I want to simulate data drift?
**Answer**: You can generate new datasets with different distributions:
```python
# Original dataset
generator = ChurnDataGenerator(n_customers=10000, churn_rate=0.15)

# Simulate drift: increased churn rate (e.g., competitor launched)
drift_generator = ChurnDataGenerator(n_customers=5000, churn_rate=0.25)

# Or modify feature distributions (e.g., more payment failures)
```

### Q5: How do I explain predictions to business stakeholders?
**Answer**: Use SHAP values to show **why** a customer is predicted to churn:
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For a specific customer
customer_idx = 42
shap.force_plot(explainer.expected_value, shap_values[customer_idx], X_test.iloc[customer_idx])

# Interpretation: "This customer is high-risk because:
#   - 3 payment failures in last 6 months (+0.25)
#   - Only 2 logins in last 30 days (+0.18)
#   - Satisfaction score of 2.5 (+0.12)"
```

---

## File Locations

- **CSV**: `data/customers.csv` (1.5 MB) - Use for pandas
- **JSON**: `data/customers.json` (9.3 MB) - Use for APIs/MongoDB
- **Summary**: `data/dataset_summary.txt` - Quick stats reference
- **Generator Script**: `data/generate_data.py` - Regenerate with different parameters

---

## Regenerating the Dataset

To create a new dataset with different parameters:

```bash
# Edit the script parameters
python data/generate_data.py

# Or modify in the script:
# generator = ChurnDataGenerator(n_customers=50000, churn_rate=0.20)
```

**Parameters you can change**:
- `n_customers`: Dataset size (default: 10,000)
- `churn_rate`: Target churn rate (default: 0.15)
- Feature distributions (age range, location list, etc.)
- Correlation strengths (churn multipliers)

---

## Next Steps

1. **Explore the data**: Use `notebooks/01_eda.ipynb` for exploratory analysis
2. **Build baseline model**: Start with logistic regression on 5 key features
3. **Feature engineering**: Create interaction features, polynomial features
4. **Model selection**: Try XGBoost, Random Forest, compare performance
5. **Threshold tuning**: Optimize for business objective (ROI)
6. **Deployment**: Package model using MLflow, create FastAPI endpoint
7. **Monitoring**: Set up drift detection using Evidently

---

## References

- **Data Generation Script**: See `data/generate_data.py` for detailed logic
- **Problem Statement**: See `problem_statement.md` for business context
- **Solution Approach**: See `solution_approach.md` for system architecture
- **Interview Questions**: See `interview_questions.md` for 20+ Q&As

---

**Last Updated**: 2025-12-15
**Dataset Version**: 1.0
**Format**: Customer churn prediction for streaming service
