"""
Sentiment Predictor for Production Inference

This module provides the main prediction interface for sentiment analysis.
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import hashlib

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

from .text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """
    Result of sentiment prediction.

    Attributes:
        sentiment: Predicted sentiment label
        confidence: Confidence score for prediction
        probabilities: Probability distribution over all classes
        processing_time_ms: Processing time in milliseconds
        model_version: Model version used
        metadata: Additional metadata
    """
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    model_version: str = "1.0.0"
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class SentimentPredictor:
    """
    Production sentiment analysis predictor.

    Features:
    - Fast inference with DistilBERT
    - Text preprocessing
    - Caching for duplicate texts
    - Batch prediction support
    - Confidence calibration

    Example:
        predictor = SentimentPredictor()
        result = predictor.predict("This product is amazing!")
        print(result.sentiment, result.confidence)
    """

    SENTIMENT_LABELS = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[str] = None,
        cache_enabled: bool = True,
        cache_size: int = 10000,
        max_length: int = 128
    ):
        """
        Initialize sentiment predictor.

        Args:
            model_name: Pretrained model name or path
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            cache_enabled: Enable result caching
            cache_size: Maximum cache size
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing SentimentPredictor on device: {self.device}")

        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()

        # Load model and tokenizer
        self._load_model()

        # Initialize cache
        self._cache: Dict[str, SentimentResult] = {}

        logger.info(f"SentimentPredictor initialized with model: {model_name}")

    def _load_model(self):
        """Load model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(
        self,
        text: str,
        preprocess: bool = True,
        use_cache: bool = True
    ) -> SentimentResult:
        """
        Predict sentiment for a single text.

        Args:
            text: Input text
            preprocess: Apply text preprocessing
            use_cache: Use cached results if available

        Returns:
            SentimentResult with prediction details

        Example:
            >>> predictor = SentimentPredictor()
            >>> result = predictor.predict("Great product!")
            >>> print(result.sentiment)
            'positive'
        """
        start_time = time.time()

        # Validate input
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        # Check cache
        if self.cache_enabled and use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_result

        # Preprocess text
        if preprocess:
            processed_text = self.preprocessor.preprocess(text)
        else:
            processed_text = text

        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        probs_list = probs.cpu().numpy()[0]

        # Get prediction
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = float(probs_list[pred_idx])

        # For binary classification (SST-2 model)
        if len(probs_list) == 2:
            sentiment = "positive" if pred_idx == 1 else "negative"
            probabilities = {
                "negative": float(probs_list[0]),
                "positive": float(probs_list[1])
            }
        else:
            # Multi-class
            sentiment = self.SENTIMENT_LABELS.get(pred_idx, "unknown")
            probabilities = {
                self.SENTIMENT_LABELS.get(i, f"class_{i}"): float(prob)
                for i, prob in enumerate(probs_list)
            }

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Create result
        result = SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=processing_time_ms,
            metadata={"original_length": len(text)}
        )

        # Cache result
        if self.cache_enabled and use_cache:
            self._add_to_cache(text, result)

        return result

    def predict_batch(
        self,
        texts: List[str],
        preprocess: bool = True,
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Predict sentiment for a batch of texts.

        Args:
            texts: List of input texts
            preprocess: Apply text preprocessing
            batch_size: Batch size for processing

        Returns:
            List of SentimentResult objects

        Example:
            >>> predictor = SentimentPredictor()
            >>> texts = ["Great!", "Terrible!", "Okay"]
            >>> results = predictor.predict_batch(texts)
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Preprocess
            if preprocess:
                processed_texts = [
                    self.preprocessor.preprocess(text) for text in batch_texts
                ]
            else:
                processed_texts = batch_texts

            # Process batch (for simplicity, using individual predictions)
            # In production, optimize with true batch processing
            for text in processed_texts:
                try:
                    result = self.predict(text, preprocess=False)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error predicting for text: {text[:50]}... Error: {e}")
                    # Add error result
                    results.append(SentimentResult(
                        sentiment="error",
                        confidence=0.0,
                        probabilities={},
                        processing_time_ms=0.0,
                        metadata={"error": str(e)}
                    ))

        return results

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _add_to_cache(self, text: str, result: SentimentResult):
        """Add result to cache."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (FIFO)
            self._cache.pop(next(iter(self._cache)))

        cache_key = self._get_cache_key(text)
        self._cache[cache_key] = result

    def clear_cache(self):
        """Clear prediction cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_enabled": self.cache_enabled
        }

    def save_model(self, path: Union[str, Path]):
        """
        Save model to disk.

        Args:
            path: Path to save directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: Union[str, Path], **kwargs) -> "SentimentPredictor":
        """
        Load model from disk.

        Args:
            path: Path to saved model directory
            **kwargs: Additional arguments for predictor

        Returns:
            Loaded SentimentPredictor instance
        """
        return cls(model_name=str(path), **kwargs)


def quick_predict(text: str) -> str:
    """
    Quick utility function for sentiment prediction.

    Args:
        text: Input text

    Returns:
        Sentiment label ('positive' or 'negative')

    Example:
        >>> quick_predict("This is amazing!")
        'positive'
    """
    predictor = SentimentPredictor()
    result = predictor.predict(text)
    return result.sentiment
