"""
Common utilities and shared components for AI/ML case studies.

This package provides reusable utilities across all case studies including:
- Data validation and schema checking
- Base model classes and interfaces
- Evaluation metrics
- Feature engineering utilities
- Monitoring and observability tools

Usage:
    from common.data_validation import validate_input, validate_schema
    from common.model_base import BaseMLModel
    from common.metrics import classification_metrics, regression_metrics
"""

from common.data_validation import (
    validate_input,
    validate_schema,
    check_missing_values,
    detect_outliers,
)
from common.model_base import BaseMLModel, ModelConfig, PredictionResult
from common.metrics import (
    classification_metrics,
    regression_metrics,
    ranking_metrics,
    MetricTracker,
)

__version__ = "0.1.0"

__all__ = [
    # Data validation
    "validate_input",
    "validate_schema",
    "check_missing_values",
    "detect_outliers",
    # Model base
    "BaseMLModel",
    "ModelConfig",
    "PredictionResult",
    # Metrics
    "classification_metrics",
    "regression_metrics",
    "ranking_metrics",
    "MetricTracker",
]
