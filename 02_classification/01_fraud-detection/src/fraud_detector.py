"""
Fraud Detection Model

This module implements the core fraud detection model with:
- Class imbalance handling (SMOTE, class weights)
- Multiple model options (Random Forest, XGBoost)
- Proper evaluation metrics for imbalanced data
- Threshold tuning

Focus: Clear, learning-oriented implementation you can explain in interviews.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict
import joblib


class FraudDetector:
    """
    Fraud Detection Model with imbalance handling.

    This class demonstrates key concepts for handling imbalanced classification:
    - SMOTE oversampling
    - Class weights
    - Threshold tuning
    - Proper evaluation metrics
    """

    def __init__(
        self,
        model_type: str = 'random_forest',
        use_smote: bool = True,
        random_state: int = 42
    ):
        """
        Initialize fraud detector.

        Parameters:
        -----------
        model_type : str
            'random_forest' or 'xgboost'
        use_smote : bool
            Whether to use SMOTE oversampling
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.random_state = random_state
        self.model = None
        self.threshold = 0.5  # Default threshold, can be tuned
        self.smote = SMOTE(random_state=random_state) if use_smote else None

    def _create_model(self):
        """
        Create the machine learning model.

        Interview Tip: Explain why tree-based models work well for fraud:
        - Handle imbalanced data with class weights
        - Interpretable (feature importance)
        - No feature scaling needed
        - Robust to outliers
        """
        if self.model_type == 'random_forest':
            # Random Forest with class weights to handle imbalance
            # class_weight='balanced' gives more weight to minority class
            model = RandomForestClassifier(
                n_estimators=100,         # Number of trees
                max_depth=10,              # Limit depth to prevent overfitting
                min_samples_split=10,      # Require at least 10 samples to split
                min_samples_leaf=5,        # Require at least 5 samples per leaf
                class_weight='balanced',   # CRITICAL: Handle class imbalance
                random_state=self.random_state,
                n_jobs=-1                  # Use all CPU cores
            )

        elif self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier

                # Calculate scale_pos_weight for XGBoost
                # This is the ratio of negative to positive samples
                # With 98% legitimate and 2% fraud, this is 49:1
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    scale_pos_weight=49,  # 98% / 2% = 49
                    random_state=self.random_state,
                    eval_metric='logloss'
                )
            except ImportError:
                print("XGBoost not installed. Falling back to Random Forest.")
                model = self._create_model_random_forest()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: bool = True
    ):
        """
        Train the fraud detection model.

        Key Steps:
        1. Optionally apply SMOTE to create synthetic fraud examples
        2. Train model with class weights
        3. Evaluate on training data (to check for overfitting)

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels (0=legitimate, 1=fraud)
        verbose : bool
            Print training progress
        """
        if verbose:
            print(f"\nTraining {self.model_type} model...")
            print(f"Original training set: {len(X_train)} samples")
            print(f"Fraud rate: {y_train.mean():.2%}")

        # Apply SMOTE if enabled
        X_resampled = X_train
        y_resampled = y_train

        if self.use_smote:
            # SMOTE creates synthetic fraud examples by interpolating
            # between existing fraud transactions
            if verbose:
                print("\nApplying SMOTE oversampling...")

            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)

            if verbose:
                print(f"After SMOTE: {len(X_resampled)} samples")
                print(f"New fraud rate: {y_resampled.mean():.2%}")

        # Create and train model
        self.model = self._create_model()

        if verbose:
            print("\nTraining model...")

        self.model.fit(X_resampled, y_resampled)

        if verbose:
            print("✓ Training complete!")

            # Show training accuracy (to check for overfitting)
            train_score = self.model.score(X_train, y_train)
            print(f"Training accuracy: {train_score:.3f}")

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict fraud (0 or 1) using custom threshold.

        Threshold tuning is CRITICAL for fraud detection:
        - Lower threshold (e.g., 0.3): Catch more fraud, more false alarms
        - Higher threshold (e.g., 0.7): Fewer false alarms, miss more fraud

        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict
        threshold : float
            Decision threshold (default: 0.5)

        Returns:
        --------
        predictions : np.ndarray
            Binary predictions (0=legitimate, 1=fraud)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if threshold is None:
            threshold = self.threshold

        # Get probability of fraud (class 1)
        probabilities = self.model.predict_proba(X)[:, 1]

        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of fraud.

        Returns probabilities instead of binary predictions.
        Useful for threshold tuning and ranking transactions by fraud risk.

        Parameters:
        -----------
        X : pd.DataFrame
            Features

        Returns:
        --------
        probabilities : np.ndarray
            Probability of fraud for each transaction
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance with proper metrics for imbalanced data.

        IMPORTANT: For imbalanced data, accuracy is misleading!
        We focus on precision, recall, F1, and PR-AUC instead.

        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            True labels
        threshold : float
            Decision threshold

        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if threshold is None:
            threshold = self.threshold

        # Get predictions and probabilities
        y_pred = self.predict(X_test, threshold=threshold)
        y_proba = self.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba)
        }

        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Threshold: {threshold:.2f}")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}  <- Can be misleading for imbalanced data!")
        print(f"  Precision: {metrics['precision']:.3f}  <- Of flagged transactions, how many are actually fraud?")
        print(f"  Recall:    {metrics['recall']:.3f}  <- Of all fraud, how much did we catch?")
        print(f"  F1 Score:  {metrics['f1']:.3f}  <- Harmonic mean of precision and recall")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}  <- Overall discrimination ability")
        print(f"  PR-AUC:    {metrics['pr_auc']:.3f}  <- Better for imbalanced data")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Legit  Fraud")
        print(f"Actual  Legit   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"        Fraud   {cm[1,0]:5d}  {cm[1,1]:5d}")
        print(f"\n  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
        print("="*50 + "\n")

        return metrics

    def tune_threshold(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_recall: float = 0.8
    ) -> float:
        """
        Find optimal threshold to achieve target recall.

        Interview Scenario: "Our business wants to catch at least 80% of fraud.
        What threshold should we use?"

        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            True labels
        target_recall : float
            Minimum desired recall (e.g., 0.8 = catch 80% of fraud)

        Returns:
        --------
        optimal_threshold : float
        """
        # Get fraud probabilities
        y_proba = self.predict_proba(X_test)

        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

        # Find threshold that achieves target recall
        # We want the highest precision while maintaining target recall
        valid_indices = np.where(recalls >= target_recall)[0]

        if len(valid_indices) == 0:
            print(f"Warning: Cannot achieve {target_recall:.0%} recall with this model")
            optimal_threshold = 0.0
        else:
            # Choose threshold with highest precision among valid recalls
            optimal_idx = valid_indices[np.argmax(precisions[valid_indices])]
            optimal_threshold = thresholds[optimal_idx]

            print(f"\nOptimal threshold for {target_recall:.0%} recall: {optimal_threshold:.3f}")
            print(f"  Precision at this threshold: {precisions[optimal_idx]:.3f}")
            print(f"  Recall at this threshold: {recalls[optimal_idx]:.3f}")

        self.threshold = optimal_threshold
        return optimal_threshold

    def plot_precision_recall_curve(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """
        Plot precision-recall curve to visualize threshold trade-offs.

        This is a KEY interview visualization showing:
        - How precision and recall trade off
        - Where to set threshold based on business needs
        """
        y_proba = self.predict_proba(X_test)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, linewidth=2)
        plt.xlabel('Recall (Fraud Detection Rate)', fontsize=12)
        plt.ylabel('Precision (Fraud Accuracy)', fontsize=12)
        plt.title('Precision-Recall Curve for Fraud Detection', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Mark current threshold
        current_precision = precision_score(y_test, self.predict(X_test))
        current_recall = recall_score(y_test, self.predict(X_test))
        plt.plot(current_recall, current_precision, 'ro', markersize=10,
                 label=f'Current threshold={self.threshold:.2f}')

        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_feature_importance(self, feature_names: list, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Critical for interviews: "How would you explain why a transaction was flagged?"

        Parameters:
        -----------
        feature_names : list
            Names of features
        top_n : int
            Number of top features to return

        Returns:
        --------
        importance_df : pd.DataFrame
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)

            return importance_df
        else:
            print("Model does not support feature importance")
            return None

    def save_model(self, filepath: str):
        """Save trained model to disk."""
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """Load trained model from disk."""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model


if __name__ == '__main__':
    """
    Example usage demonstrating the full workflow.
    """
    from data_preprocessor import load_and_prepare_data

    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data('data/transactions.csv')

    # Create and train fraud detector
    detector = FraudDetector(
        model_type='random_forest',
        use_smote=True,
        random_state=42
    )

    detector.train(X_train, y_train)

    # Evaluate with default threshold
    print("\nEvaluating with default threshold (0.5):")
    detector.evaluate(X_test, y_test, threshold=0.5)

    # Tune threshold for 80% recall
    print("\nTuning threshold for 80% recall:")
    detector.tune_threshold(X_test, y_test, target_recall=0.8)

    # Re-evaluate with tuned threshold
    detector.evaluate(X_test, y_test)

    # Show feature importance
    print("\nTop 10 most important features:")
    importance = detector.get_feature_importance(X_train.columns)
    print(importance)
