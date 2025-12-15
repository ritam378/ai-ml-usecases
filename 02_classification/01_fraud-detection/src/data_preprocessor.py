"""
Data Preprocessing for Fraud Detection

This module handles data loading, cleaning, and preprocessing for fraud detection.
Focus: Learning-oriented implementation with clear explanations.

Key Learning Points:
- How to handle imbalanced data splitting (stratified split)
- When and why to scale features
- Avoiding data leakage in preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class FraudDataPreprocessor:
    """
    Preprocessor for fraud detection data.

    Simple, clear implementation showing proper preprocessing steps.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize preprocessor.

        Parameters:
        -----------
        test_size : float
            Proportion of data for testing (default 0.2 = 20%)
        random_state : int
            Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load transaction data from CSV or JSON.

        Parameters:
        -----------
        filepath : str
            Path to data file (supports .csv or .json)

        Returns:
        --------
        pd.DataFrame
            Loaded transaction data
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}. Use .csv or .json")

        print(f"Loaded {len(df)} transactions from {filepath}")
        print(f"Fraud rate: {df['Class'].mean():.2%}")

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        scale_features: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.

        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset with features and 'Class' column
        scale_features : bool
            Whether to scale features (not needed for tree-based models)

        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (0=legitimate, 1=fraud)
        """
        # Separate features (X) and target (y)
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Store feature names for later reference
        self.feature_names = X.columns.tolist()

        # Optional: Scale features (for models like logistic regression, neural networks)
        # Tree-based models (Random Forest, XGBoost) don't need scaling
        if scale_features:
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            print("Features scaled using StandardScaler")

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.

        IMPORTANT: Always split BEFORE applying any resampling techniques.
        This prevents data leakage where test data influences training.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        stratify : bool
            If True, maintains class distribution in train/test splits.
            CRITICAL for imbalanced data to ensure both sets have fraud examples.

        Returns:
        --------
        X_train, X_test, y_train, y_test
            Train and test splits
        """
        # Stratified split maintains the fraud ratio in both train and test
        # With 2% fraud rate, this ensures test set also has ~2% fraud
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if stratify else None  # Stratify by class labels
        )

        # Print split statistics
        print(f"\nTrain set: {len(X_train)} transactions ({y_train.mean():.2%} fraud)")
        print(f"Test set: {len(X_test)} transactions ({y_test.mean():.2%} fraud)")

        # Verify stratification worked
        if stratify:
            train_fraud_rate = y_train.mean()
            test_fraud_rate = y_test.mean()
            overall_fraud_rate = y.mean()

            # Rates should be very similar (within 0.5%)
            assert abs(train_fraud_rate - overall_fraud_rate) < 0.005, \
                "Stratification failed: train fraud rate differs significantly"
            assert abs(test_fraud_rate - overall_fraud_rate) < 0.005, \
                "Stratification failed: test fraud rate differs significantly"

        return X_train, X_test, y_train, y_test

    def get_class_distribution(self, y: pd.Series) -> dict:
        """
        Get class distribution statistics.

        Useful for understanding class imbalance and choosing appropriate techniques.

        Parameters:
        -----------
        y : pd.Series
            Target variable

        Returns:
        --------
        dict
            Class distribution statistics
        """
        total = len(y)
        n_fraud = (y == 1).sum()
        n_legitimate = (y == 0).sum()

        stats = {
            'total': total,
            'n_legitimate': n_legitimate,
            'n_fraud': n_fraud,
            'fraud_ratio': n_fraud / total,
            'legitimate_ratio': n_legitimate / total,
            'imbalance_ratio': n_legitimate / n_fraud if n_fraud > 0 else float('inf')
        }

        return stats

    def print_data_summary(self, X: pd.DataFrame, y: pd.Series):
        """
        Print helpful data summary for learning/debugging.

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        """
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)

        # Class distribution
        dist = self.get_class_distribution(y)
        print(f"\nClass Distribution:")
        print(f"  Legitimate: {dist['n_legitimate']:,} ({dist['legitimate_ratio']:.1%})")
        print(f"  Fraud: {dist['n_fraud']:,} ({dist['fraud_ratio']:.1%})")
        print(f"  Imbalance Ratio: {dist['imbalance_ratio']:.1f}:1")

        # Feature statistics
        print(f"\nFeatures: {len(X.columns)}")
        print(f"  Numeric features: {X.select_dtypes(include=[np.number]).shape[1]}")

        # Missing values
        missing = X.isnull().sum().sum()
        print(f"\nMissing Values: {missing}")

        # Memory usage
        print(f"\nDataset Size: {X.memory_usage(deep=True).sum() / 1024:.1f} KB")

        print("="*50 + "\n")


def load_and_prepare_data(
    filepath: str = 'data/transactions.csv',
    test_size: float = 0.2,
    scale_features: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function to load and prepare data in one step.

    This is the typical workflow:
    1. Load data
    2. Prepare features (X) and target (y)
    3. Split into train/test with stratification

    Parameters:
    -----------
    filepath : str
        Path to transaction data
    test_size : float
        Test set proportion
    scale_features : bool
        Whether to scale features
    random_state : int
        Random seed

    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    # Initialize preprocessor
    preprocessor = FraudDataPreprocessor(
        test_size=test_size,
        random_state=random_state
    )

    # Load data
    df = preprocessor.load_data(filepath)

    # Prepare features and target
    X, y = preprocessor.prepare_features(df, scale_features=scale_features)

    # Print summary
    preprocessor.print_data_summary(X, y)

    # Split data (stratified to maintain fraud ratio)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, stratify=True)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    """
    Example usage for learning/testing.
    """
    print("Loading and preprocessing fraud detection data...")

    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        filepath='data/transactions.csv',
        scale_features=False  # Tree-based models don't need scaling
    )

    print("\nPreprocessing complete!")
    print(f"Training set ready: {X_train.shape}")
    print(f"Test set ready: {X_test.shape}")
