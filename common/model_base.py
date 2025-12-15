"""
Base model classes and interfaces for ML models.

This module provides abstract base classes and common interfaces for all ML models
in the case studies, ensuring consistency and reusability.

Usage:
    from common.model_base import BaseMLModel, ModelConfig, PredictionResult

    class MyModel(BaseMLModel):
        def fit(self, X, y):
            # Training logic
            pass

        def predict(self, X):
            # Prediction logic
            pass
"""

import hashlib
import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for ML models.

    Attributes:
        model_name: Name identifier for the model
        model_type: Type of model (e.g., "classifier", "regressor", "llm")
        version: Model version string
        hyperparameters: Dict of model hyperparameters
        metadata: Additional metadata
        created_at: Timestamp of model creation
    """
    model_name: str
    model_type: str
    version: str = "1.0.0"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "hyperparameters": self.hyperparameters,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**data)

    def get_cache_key(self) -> str:
        """
        Generate unique cache key based on config.

        Returns:
            MD5 hash of serialized config
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


@dataclass
class PredictionResult:
    """
    Result of model prediction.

    Attributes:
        predictions: Model predictions (can be labels, probabilities, etc.)
        probabilities: Prediction probabilities (for classification)
        confidence: Confidence scores for predictions
        latency_ms: Prediction latency in milliseconds
        metadata: Additional metadata (model version, features used, etc.)
        timestamp: Timestamp of prediction
    """
    predictions: Union[np.ndarray, List[Any]]
    probabilities: Optional[np.ndarray] = None
    confidence: Optional[Union[float, np.ndarray]] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "predictions": (
                self.predictions.tolist()
                if isinstance(self.predictions, np.ndarray)
                else self.predictions
            ),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

        if self.probabilities is not None:
            result["probabilities"] = self.probabilities.tolist()

        if self.confidence is not None:
            result["confidence"] = (
                self.confidence.tolist()
                if isinstance(self.confidence, np.ndarray)
                else self.confidence
            )

        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms

        return result


class BaseMLModel(ABC):
    """
    Abstract base class for all ML models.

    This class provides common functionality including:
    - Model saving/loading
    - Versioning
    - Metadata tracking
    - Prediction interface
    - Performance monitoring

    Attributes:
        config: Model configuration
        model: The underlying ML model (sklearn, PyTorch, etc.)
        is_fitted: Whether model has been trained
        metrics: Training/evaluation metrics
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize base model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Optional[Any] = None
        self.is_fitted: bool = False
        self.metrics: Dict[str, float] = {}
        self._feature_names: Optional[List[str]] = None

    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **kwargs) -> "BaseMLModel":
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels (None for unsupervised)
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: Any, **kwargs) -> PredictionResult:
        """
        Make predictions.

        Args:
            X: Input features
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult with predictions and metadata
        """
        pass

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model file

        Example:
            model.save("models/my_model.pkl")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "config": self.config.to_dict(),
            "model": self.model,
            "is_fitted": self.is_fitted,
            "metrics": self.metrics,
            "feature_names": self._feature_names,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseMLModel":
        """
        Load model from disk.

        Args:
            path: Path to model file

        Returns:
            Loaded model instance

        Example:
            model = MyModel.load("models/my_model.pkl")
        """
        path = Path(path)

        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        config = ModelConfig.from_dict(save_dict["config"])
        instance = cls(config)

        instance.model = save_dict["model"]
        instance.is_fitted = save_dict["is_fitted"]
        instance.metrics = save_dict["metrics"]
        instance._feature_names = save_dict.get("feature_names")

        logger.info(f"Model loaded from {path}")
        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model details
        """
        return {
            "config": self.config.to_dict(),
            "is_fitted": self.is_fitted,
            "metrics": self.metrics,
            "feature_names": self._feature_names,
        }

    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set feature names for the model.

        Args:
            feature_names: List of feature names
        """
        self._feature_names = feature_names

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get feature names.

        Returns:
            List of feature names or None
        """
        return self._feature_names

    def _validate_fitted(self) -> None:
        """
        Check if model is fitted before prediction.

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Model {self.config.model_name} is not fitted. "
                f"Call fit() before predict()."
            )

    def _measure_latency(self, func, *args, **kwargs):
        """
        Measure function execution latency.

        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, latency_ms)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000
        return result, latency_ms


class EnsembleModel(BaseMLModel):
    """
    Base class for ensemble models combining multiple models.

    Attributes:
        models: List of base models in the ensemble
        weights: Optional weights for weighted voting/averaging
    """

    def __init__(
        self,
        config: ModelConfig,
        models: Optional[List[BaseMLModel]] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble model.

        Args:
            config: Model configuration
            models: List of base models
            weights: Optional weights for each model
        """
        super().__init__(config)
        self.models = models or []
        self.weights = weights

        if self.weights and len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

    def add_model(self, model: BaseMLModel, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.

        Args:
            model: Model to add
            weight: Weight for this model's predictions
        """
        self.models.append(model)

        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs) -> "EnsembleModel":
        """
        Train all models in the ensemble.

        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining
        """
        logger.info(f"Training ensemble with {len(self.models)} models")

        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            model.fit(X, y, **kwargs)

        self.is_fitted = True
        return self

    @abstractmethod
    def predict(self, X: Any, **kwargs) -> PredictionResult:
        """
        Make ensemble predictions.

        Implementation should combine predictions from all models.

        Args:
            X: Input features
            **kwargs: Additional prediction parameters

        Returns:
            Combined PredictionResult
        """
        pass
