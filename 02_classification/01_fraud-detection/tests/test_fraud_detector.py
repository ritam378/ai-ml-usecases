"""
Tests for FraudDetector class.

Focus: Key functionality and learning concepts.
"""

import pytest
import numpy as np
from src.fraud_detector import FraudDetector


class TestFraudDetector:
    """Test fraud detection model."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        detector = FraudDetector(model_type='random_forest', use_smote=True)

        assert detector.model_type == 'random_forest'
        assert detector.use_smote is True
        assert detector.threshold == 0.5
        assert detector.model is None  # Not trained yet

    def test_training(self, train_test_split_data):
        """Test model training completes successfully."""
        X_train, X_test, y_train, y_test = train_test_split_data

        detector = FraudDetector(model_type='random_forest', use_smote=False)
        detector.train(X_train, y_train, verbose=False)

        # Check model was created
        assert detector.model is not None

        # Check model can make predictions
        predictions = detector.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})  # Only 0 and 1

    def test_smote_oversampling(self, train_test_split_data):
        """Test SMOTE increases minority class samples."""
        X_train, X_test, y_train, y_test = train_test_split_data

        # Train with SMOTE
        detector_with_smote = FraudDetector(use_smote=True)
        detector_with_smote.train(X_train, y_train, verbose=False)

        # Train without SMOTE
        detector_no_smote = FraudDetector(use_smote=False)
        detector_no_smote.train(X_train, y_train, verbose=False)

        # Both should work
        assert detector_with_smote.model is not None
        assert detector_no_smote.model is not None

    def test_predict_proba(self, train_test_split_data):
        """Test probability predictions."""
        X_train, X_test, y_train, y_test = train_test_split_data

        detector = FraudDetector(use_smote=False)
        detector.train(X_train, y_train, verbose=False)

        probabilities = detector.predict_proba(X_test)

        # Check probabilities are valid
        assert len(probabilities) == len(X_test)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_threshold_tuning(self, train_test_split_data):
        """Test custom threshold changes predictions."""
        X_train, X_test, y_train, y_test = train_test_split_data

        detector = FraudDetector(use_smote=False)
        detector.train(X_train, y_train, verbose=False)

        # Low threshold should predict more fraud
        pred_low = detector.predict(X_test, threshold=0.3)
        pred_high = detector.predict(X_test, threshold=0.7)

        # Lower threshold should predict more positives (fraud)
        assert pred_low.sum() >= pred_high.sum()

    def test_evaluation_metrics(self, train_test_split_data):
        """Test evaluation returns expected metrics."""
        X_train, X_test, y_train, y_test = train_test_split_data

        detector = FraudDetector(use_smote=False)
        detector.train(X_train, y_train, verbose=False)

        metrics = detector.evaluate(X_test, y_test, threshold=0.5)

        # Check all expected metrics are present
        expected_keys = ['threshold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        for key in expected_keys:
            assert key in metrics

        # Check metrics are in valid ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1

    def test_model_requires_training(self, sample_features_and_target):
        """Test that prediction fails without training."""
        X, y = sample_features_and_target

        detector = FraudDetector()

        # Should raise error before training
        with pytest.raises(ValueError, match="Model not trained"):
            detector.predict(X)

        with pytest.raises(ValueError, match="Model not trained"):
            detector.predict_proba(X)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
