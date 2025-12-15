"""
Generate synthetic credit card fraud detection dataset.

This script creates a realistic but synthetic dataset for learning purposes.
The dataset mimics the structure of real fraud detection datasets with:
- Severe class imbalance (~2% fraud rate)
- PCA-transformed features (V1-V28) to simulate anonymized data
- Time and Amount features
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import json


def generate_fraud_dataset(n_samples=1000, fraud_ratio=0.02, random_state=42):
    """
    Generate synthetic fraud detection dataset.

    Parameters:
    -----------
    n_samples : int
        Total number of transactions
    fraud_ratio : float
        Proportion of fraudulent transactions (0.02 = 2%)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Synthetic transaction dataset
    """
    np.random.seed(random_state)

    # Calculate class distribution
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud

    # Generate base features using sklearn's make_classification
    # This creates separable but overlapping classes
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,  # Will map to Time, Amount, and V1-V28
        n_informative=20,  # 20 features actually useful for classification
        n_redundant=5,     # 5 features are combinations of informative
        n_classes=2,
        weights=[1-fraud_ratio, fraud_ratio],  # Class imbalance
        flip_y=0.01,  # 1% label noise (simulates mislabeled transactions)
        random_state=random_state
    )

    # Create DataFrame
    df = pd.DataFrame()

    # Time: Seconds elapsed from first transaction
    # Normal pattern: transactions throughout the day
    # Fraud pattern: Often happens in bursts (unusual timing)
    time_normal = np.random.uniform(0, 172800, n_legitimate)  # 2 days in seconds
    # Split fraud times: half late night, half clustered
    time_fraud_night = np.random.uniform(0, 7200, n_fraud // 2)      # Late night (suspicious)
    time_fraud_burst = np.random.uniform(50000, 60000, (n_fraud - n_fraud // 2))  # Clustered in time (burst)
    time_fraud = np.concatenate([time_fraud_night, time_fraud_burst])

    df['Time'] = np.concatenate([time_normal, time_fraud])

    # Amount: Transaction value
    # Normal: Typical purchases ($10-$500)
    # Fraud: Often higher amounts or very specific amounts
    amount_normal = np.random.gamma(shape=2, scale=50, size=n_legitimate)
    amount_fraud = np.concatenate([
        np.random.gamma(shape=5, scale=100, size=n_fraud // 2),  # High value fraud
        np.random.uniform(1, 10, n_fraud // 2)  # Low value testing
    ])

    df['Amount'] = np.concatenate([amount_normal, amount_fraud])

    # V1-V28: PCA-transformed anonymized features
    # In reality, these might be:
    # - Geographic features (distance from home, unusual location)
    # - Merchant features (merchant category, reputation)
    # - Behavioral features (transaction velocity, typing speed)
    # - Device features (device fingerprint, IP address)

    for i in range(28):
        df[f'V{i+1}'] = X[:, i+2] if i+2 < X.shape[1] else np.random.randn(n_samples)

    # Add some feature engineering to make patterns more realistic
    # Fraud transactions often have:
    # - Unusual combinations of features
    # - Outlier values in certain dimensions

    fraud_idx = n_legitimate + np.arange(n_fraud)
    for idx in fraud_idx:
        # Make some V features more extreme for fraud
        extreme_features = np.random.choice(28, size=5, replace=False)
        for feat in extreme_features:
            df.loc[idx, f'V{feat+1}'] *= np.random.uniform(2, 4)

    # Target variable
    df['Class'] = np.concatenate([np.zeros(n_legitimate), np.ones(n_fraud)])

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Round values for cleaner display
    df['Time'] = df['Time'].round(2)
    df['Amount'] = df['Amount'].round(2)
    for i in range(1, 29):
        df[f'V{i}'] = df[f'V{i}'].round(6)
    df['Class'] = df['Class'].astype(int)

    return df


def save_dataset(df, output_dir='data'):
    """Save dataset in CSV and JSON formats."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save as CSV
    csv_path = os.path.join(output_dir, 'transactions.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save as JSON
    json_path = os.path.join(output_dir, 'transactions.json')
    df.to_json(json_path, orient='records', indent=2)
    print(f"Saved JSON: {json_path}")

    # Print dataset statistics
    print(f"\n--- Dataset Statistics ---")
    print(f"Total transactions: {len(df)}")
    print(f"Legitimate transactions: {(df['Class'] == 0).sum()} ({(df['Class'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"Fraudulent transactions: {(df['Class'] == 1).sum()} ({(df['Class'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"\nAmount statistics:")
    print(df.groupby('Class')['Amount'].describe())


if __name__ == '__main__':
    # Generate dataset
    print("Generating synthetic fraud detection dataset...")
    df = generate_fraud_dataset(n_samples=1000, fraud_ratio=0.02)

    # Save to files
    save_dataset(df)

    print("\n✓ Dataset generation complete!")
    print("\nExample transactions (first 5 rows):")
    print(df.head())
