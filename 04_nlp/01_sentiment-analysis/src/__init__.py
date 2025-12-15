"""
Sentiment Analysis at Scale - Source Code

This package provides production-ready sentiment analysis using DistilBERT.

Modules:
    text_preprocessor: Text cleaning and normalization
    model_trainer: Model training and fine-tuning
    sentiment_predictor: Inference and prediction
    data_generator: Synthetic data generation for testing
"""

__version__ = "1.0.0"

from .text_preprocessor import TextPreprocessor
from .sentiment_predictor import SentimentPredictor

__all__ = [
    "TextPreprocessor",
    "SentimentPredictor",
]
