"""
Pytest fixtures for fraud detection tests.

Provides reusable test data and components.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@pytest.fixture
def sample_fraud_data():
    """
    Create small synthetic fraud dataset for testing.

    Returns balanced classes for easier testing (not realistic but good for unit tests).
    """
    # Create small balanced dataset for testing
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=2,
        weights=[0.8, 0.2],  # 20% fraud
        random_state=42
    )

    # Convert to DataFrame
    feature_names = [f'V{i+1}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Time'] = np.random.uniform(0, 10000, 100)
    df['Amount'] = np.random.uniform(1, 500, 100)
    df['Class'] = y

    return df


@pytest.fixture
def sample_features_and_target(sample_fraud_data):
    """
    Split sample data into features and target.
    """
    X = sample_fraud_data.drop('Class', axis=1)
    y = sample_fraud_data['Class']
    return X, y


@pytest.fixture
def train_test_split_data(sample_features_and_target):
    """
    Pre-split data for testing.
    """
    from sklearn.model_selection import train_test_split

    X, y = sample_features_and_target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
