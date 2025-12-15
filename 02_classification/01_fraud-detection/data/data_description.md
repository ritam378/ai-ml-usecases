# Data Description: Credit Card Fraud Detection Dataset

## Overview

This is a synthetic dataset created for learning fraud detection concepts. It mimics the structure and characteristics of real-world fraud detection datasets while being safe for educational use.

## Dataset Statistics

- **Total Transactions**: 1,000
- **Legitimate Transactions**: 980 (98.0%)
- **Fraudulent Transactions**: 20 (2.0%)
- **Class Imbalance Ratio**: 49:1 (severe imbalance)

## Features

### Time
- **Type**: Continuous (float)
- **Range**: 0 to 172,800 seconds (2 days)
- **Description**: Number of seconds elapsed between this transaction and the first transaction in the dataset
- **Fraud Pattern**: Fraudulent transactions tend to cluster in time (bursts) or occur at unusual hours (late night)

### Amount
- **Type**: Continuous (float)
- **Range**: $0.01 to $1,500
- **Description**: Transaction amount in dollars
- **Distribution**:
  - **Legitimate**: Mean ~$103, typically $10-$500 (gamma distribution)
  - **Fraudulent**: Mean ~$287, either very high ($300-$1,300) or very low ($1-$10)
- **Fraud Pattern**: Fraudsters often test with small amounts first, then attempt large fraudulent transactions

### V1 through V28
- **Type**: Continuous (float)
- **Description**: Principal Component Analysis (PCA) transformed features
- **Why PCA?**: In real-world datasets, these features are anonymized to protect user privacy
- **Real-World Examples** (what these might represent before PCA):
  - Geographic features (distance from home, country, etc.)
  - Merchant category codes
  - Transaction velocity (transactions per hour)
  - Device fingerprinting
  - Time since last transaction
  - Cardholder demographics
  - Historical spending patterns
  - IP address information

- **Fraud Pattern**: Fraudulent transactions have more extreme values in certain V features (outliers)

### Class (Target Variable)
- **Type**: Binary (integer)
- **Values**:
  - `0` = Legitimate transaction (98%)
  - `1` = Fraudulent transaction (2%)
- **Description**: Ground truth label for model training and evaluation

## File Formats

### transactions.csv
Standard CSV format with headers. Easy to load with pandas:

```python
import pandas as pd
df = pd.read_csv('data/transactions.csv')
```

### transactions.json
JSON array format with each transaction as an object:

```python
import pandas as pd
df = pd.read_json('data/transactions.json')
```

## Data Generation

The dataset is generated using [generate_data.py](generate_data.py) with:
- `sklearn.datasets.make_classification` for base separable classes
- Custom logic to inject fraud patterns
- Controlled random seed (42) for reproducibility

## Key Characteristics for Learning

### 1. Severe Class Imbalance
This is the **primary learning challenge**:
- 98% legitimate, 2% fraud
- A model predicting "all legitimate" achieves 98% accuracy but catches 0% fraud
- Demonstrates why accuracy is a poor metric
- Requires special handling: SMOTE, class weights, custom thresholds

### 2. Realistic Patterns
The data includes realistic fraud indicators:
- **Amount patterns**: Fraudsters test with small amounts
- **Time patterns**: Fraud often happens in bursts or at unusual times
- **Feature distributions**: Fraud has outlier feature values
- **Label noise**: 1% mislabeled transactions (mimics real-world ambiguity)

### 3. Manageable Size
- 1,000 transactions (vs millions in production)
- Fast training and iteration
- Easy to inspect and debug
- Perfect for learning and experimentation

## Usage Examples

### Load Data

```python
import pandas as pd

# Load CSV
df = pd.read_csv('data/transactions.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Total transactions: {len(df)}")
print(f"Fraud rate: {y.mean():.2%}")
```

### Check Class Distribution

```python
import matplotlib.pyplot as plt

# Count plot
class_counts = df['Class'].value_counts()
print(class_counts)

# Visualize
plt.bar(['Legitimate', 'Fraud'], class_counts.values)
plt.ylabel('Count')
plt.title('Class Distribution (Severe Imbalance)')
plt.show()
```

### Explore Feature Differences

```python
import seaborn as sns

# Compare Amount distribution by class
sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Amount Distribution: Legitimate vs Fraud')
plt.show()

# Summary statistics by class
print(df.groupby('Class').describe())
```

## Interview Discussion Points

When discussing this dataset in interviews:

1. **Class Imbalance**: "The 98:2 ratio is realistic for fraud detection. Most transactions are legitimate, but we need to find the 2% fraud without blocking legitimate customers."

2. **Feature Engineering**: "In production, I'd create additional features like transaction velocity, geographic distance, merchant risk scores. Here, V1-V28 are PCA'd to simulate that."

3. **Data Privacy**: "Real fraud detection datasets can't be shared due to PCI-DSS compliance. PCA transformation helps anonymize sensitive information while preserving predictive power."

4. **Evaluation Strategy**: "I'd use stratified train-test split to maintain the 98:2 ratio in both sets. Then evaluate with precision, recall, F1, and PR-AUC instead of accuracy."

5. **Data Quality**: "I added 1% label noise to simulate real-world challenges where some transactions are disputed or mislabeled. The model needs to be robust to this."

## Regenerating the Dataset

To create a new dataset with different parameters:

```bash
python3 data/generate_data.py
```

Or modify in the script:
```python
df = generate_fraud_dataset(
    n_samples=2000,      # More transactions
    fraud_ratio=0.05,    # 5% fraud rate
    random_state=123     # Different random seed
)
```

## Comparison to Real-World Datasets

| Aspect | This Dataset | Real-World (e.g., Kaggle Credit Card) |
|--------|-------------|--------------------------------------|
| Size | 1,000 transactions | 284,807 transactions |
| Fraud Rate | 2% | 0.17% (more imbalanced!) |
| Features | 30 (Time, Amount, V1-V28) | Same structure |
| PCA | Simulated | Actual PCA for privacy |
| Patterns | Synthetic but realistic | Real fraud patterns |
| Purpose | Learning | Research/Kaggle competition |

## License & Usage

This is synthetic data generated for educational purposes. Free to use for:
- Learning ML concepts
- Interview preparation
- Academic projects
- Portfolio demonstrations

Not suitable for:
- Production fraud detection
- Academic research claiming real-world results
- Benchmarking fraud detection systems
