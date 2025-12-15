"""
Tests for Sentiment Predictor Module

Tests cover:
- Model initialization
- Prediction (single and batch)
- Caching mechanism
- Result formatting
- Error handling
- Model save/load
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from src.sentiment_predictor import (
    SentimentPredictor,
    SentimentResult,
    quick_predict
)


class TestSentimentResult:
    """Test cases for SentimentResult dataclass."""

    def test_creation(self):
        """Test basic result creation."""
        result = SentimentResult(
            sentiment="positive",
            confidence=0.95,
            probabilities={"positive": 0.95, "negative": 0.05},
            processing_time_ms=10.5
        )
        assert result.sentiment == "positive"
        assert result.confidence == 0.95
        assert result.processing_time_ms == 10.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SentimentResult(
            sentiment="positive",
            confidence=0.95,
            probabilities={"positive": 0.95, "negative": 0.05},
            processing_time_ms=10.5
        )
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["sentiment"] == "positive"
        assert result_dict["confidence"] == 0.95
        assert "timestamp" in result_dict
        assert "model_version" in result_dict

    def test_default_values(self):
        """Test default values in result."""
        result = SentimentResult(
            sentiment="positive",
            confidence=0.95,
            probabilities={},
            processing_time_ms=10.0
        )
        assert result.model_version == "1.0.0"
        assert result.metadata == {}
        assert result.timestamp is not None


@pytest.mark.model
class TestSentimentPredictorInitialization:
    """Test cases for predictor initialization."""

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_initialization_default(self, mock_model, mock_tokenizer):
        """Test default initialization."""
        predictor = SentimentPredictor()

        assert predictor.device in ["cpu", "cuda"]
        assert predictor.cache_enabled is True
        assert predictor.cache_size == 10000
        assert predictor.max_length == 128

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_initialization_custom_device(self, mock_model, mock_tokenizer):
        """Test initialization with custom device."""
        predictor = SentimentPredictor(device="cpu")
        assert predictor.device == "cpu"

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_initialization_cache_disabled(self, mock_model, mock_tokenizer):
        """Test initialization with cache disabled."""
        predictor = SentimentPredictor(cache_enabled=False)
        assert predictor.cache_enabled is False

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_initialization_custom_params(self, mock_model, mock_tokenizer):
        """Test initialization with custom parameters."""
        predictor = SentimentPredictor(
            cache_size=5000,
            max_length=256
        )
        assert predictor.cache_size == 5000
        assert predictor.max_length == 256


@pytest.mark.model
class TestSentimentPredictorPrediction:
    """Test cases for prediction functionality."""

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_valid_text(self, mock_model_class, mock_tokenizer_class):
        """Test prediction with valid text."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock model output for positive sentiment
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.2, 0.8]])  # Binary: negative, positive
        mock_model.return_value = mock_output

        # Create predictor and predict
        predictor = SentimentPredictor()
        result = predictor.predict("This is great!")

        assert isinstance(result, SentimentResult)
        assert result.sentiment in ["positive", "negative"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time_ms >= 0

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_empty_text(self, mock_model, mock_tokenizer):
        """Test prediction with empty text."""
        predictor = SentimentPredictor()

        with pytest.raises(ValueError, match="non-empty string"):
            predictor.predict("")

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_none_text(self, mock_model, mock_tokenizer):
        """Test prediction with None text."""
        predictor = SentimentPredictor()

        with pytest.raises(ValueError, match="non-empty string"):
            predictor.predict(None)

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_invalid_type(self, mock_model, mock_tokenizer):
        """Test prediction with invalid type."""
        predictor = SentimentPredictor()

        with pytest.raises(ValueError, match="non-empty string"):
            predictor.predict(123)

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_with_preprocessing(self, mock_model_class, mock_tokenizer_class):
        """Test prediction with preprocessing enabled."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictor = SentimentPredictor()
        result = predictor.predict("GREAT!!! http://example.com", preprocess=True)

        assert isinstance(result, SentimentResult)

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_without_preprocessing(self, mock_model_class, mock_tokenizer_class):
        """Test prediction without preprocessing."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictor = SentimentPredictor()
        result = predictor.predict("great", preprocess=False)

        assert isinstance(result, SentimentResult)


@pytest.mark.model
class TestSentimentPredictorCaching:
    """Test cases for caching functionality."""

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_cache_enabled(self, mock_model_class, mock_tokenizer_class):
        """Test that caching works."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictor = SentimentPredictor(cache_enabled=True)

        # First prediction
        result1 = predictor.predict("test text")

        # Second prediction (should hit cache)
        result2 = predictor.predict("test text")

        # Should call model only once
        assert mock_model.call_count == 1
        assert result1.sentiment == result2.sentiment

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_cache_disabled(self, mock_model_class, mock_tokenizer_class):
        """Test prediction without caching."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictor = SentimentPredictor(cache_enabled=False)

        # Two predictions
        predictor.predict("test text")
        predictor.predict("test text")

        # Should call model twice
        assert mock_model.call_count == 2

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_clear_cache(self, mock_model, mock_tokenizer):
        """Test cache clearing."""
        predictor = SentimentPredictor()
        predictor._cache = {"key": "value"}

        predictor.clear_cache()
        assert len(predictor._cache) == 0

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_cache_stats(self, mock_model, mock_tokenizer):
        """Test cache statistics."""
        predictor = SentimentPredictor(cache_size=1000, cache_enabled=True)

        stats = predictor.get_cache_stats()

        assert stats["cache_enabled"] is True
        assert stats["max_cache_size"] == 1000
        assert "cache_size" in stats


@pytest.mark.model
class TestSentimentPredictorBatch:
    """Test cases for batch prediction."""

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_batch_valid(self, mock_model_class, mock_tokenizer_class):
        """Test batch prediction with valid texts."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictor = SentimentPredictor()
        texts = ["Great product!", "Terrible service", "It's okay"]

        results = predictor.predict_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_batch_empty(self, mock_model, mock_tokenizer):
        """Test batch prediction with empty list."""
        predictor = SentimentPredictor()
        results = predictor.predict_batch([])

        assert len(results) == 0

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_predict_batch_with_batch_size(self, mock_model_class, mock_tokenizer_class):
        """Test batch prediction with custom batch size."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictor = SentimentPredictor()
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

        results = predictor.predict_batch(texts, batch_size=2)

        assert len(results) == 5


@pytest.mark.model
class TestSentimentPredictorSaveLoad:
    """Test cases for model save/load functionality."""

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_save_model(self, mock_model_class, mock_tokenizer_class, temp_dir):
        """Test model saving."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        predictor = SentimentPredictor()
        save_path = temp_dir / "model"

        predictor.save_model(save_path)

        # Should call save_pretrained on both model and tokenizer
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_load_model(self, mock_model_class, mock_tokenizer_class):
        """Test model loading."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        predictor = SentimentPredictor.load_model("path/to/model", device="cpu")

        assert predictor.device == "cpu"


@pytest.mark.model
class TestQuickPredict:
    """Test utility function."""

    @patch('src.sentiment_predictor.SentimentPredictor')
    def test_quick_predict(self, mock_predictor_class):
        """Test quick_predict utility function."""
        mock_predictor = MagicMock()
        mock_result = MagicMock()
        mock_result.sentiment = "positive"
        mock_predictor.predict.return_value = mock_result
        mock_predictor_class.return_value = mock_predictor

        result = quick_predict("Great!")

        assert result == "positive"
        mock_predictor.predict.assert_called_once_with("Great!")


class TestSentimentPredictorEdgeCases:
    """Test edge cases and error handling."""

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_very_long_text(self, mock_model_class, mock_tokenizer_class):
        """Test prediction with very long text."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1] * 128]),
            'attention_mask': torch.tensor([[1] * 128])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictor = SentimentPredictor()
        long_text = "word " * 1000

        result = predictor.predict(long_text)
        assert isinstance(result, SentimentResult)

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_cache_key_generation(self, mock_model, mock_tokenizer):
        """Test cache key generation is consistent."""
        predictor = SentimentPredictor()

        key1 = predictor._get_cache_key("test text")
        key2 = predictor._get_cache_key("test text")

        assert key1 == key2
        assert isinstance(key1, str)

    @patch('src.sentiment_predictor.AutoTokenizer')
    @patch('src.sentiment_predictor.AutoModelForSequenceClassification')
    def test_cache_overflow(self, mock_model_class, mock_tokenizer_class):
        """Test cache overflow handling."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output

        predictor = SentimentPredictor(cache_size=2, cache_enabled=True)

        # Add 3 items (should evict first)
        predictor.predict("text 1")
        predictor.predict("text 2")
        predictor.predict("text 3")

        stats = predictor.get_cache_stats()
        assert stats["cache_size"] <= 2
