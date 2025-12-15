"""
Tests for Text Preprocessor Module

Tests cover:
- Text preprocessing pipeline
- Individual preprocessing steps
- Feature extraction
- Text validation
- Edge cases
"""

import pytest
from src.text_preprocessor import TextPreprocessor, clean_review_text


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def test_initialization_default(self):
        """Test default initialization."""
        preprocessor = TextPreprocessor()
        assert preprocessor.lowercase is True
        assert preprocessor.remove_urls is True
        assert preprocessor.remove_html is True
        assert preprocessor.normalize_whitespace is True
        assert preprocessor.normalize_repeats is True
        assert preprocessor.max_repeat == 3

    def test_initialization_custom(self):
        """Test custom initialization."""
        preprocessor = TextPreprocessor(
            lowercase=False,
            remove_urls=False,
            max_repeat=2
        )
        assert preprocessor.lowercase is False
        assert preprocessor.remove_urls is False
        assert preprocessor.max_repeat == 2

    def test_preprocess_basic(self):
        """Test basic preprocessing."""
        preprocessor = TextPreprocessor()
        text = "This is a SIMPLE test."
        result = preprocessor.preprocess(text)
        assert result == "this is a simple test."

    def test_remove_urls(self):
        """Test URL removal."""
        preprocessor = TextPreprocessor()
        text = "Check this: http://example.com and https://test.com/path"
        result = preprocessor.preprocess(text)
        assert "http://" not in result
        assert "https://" not in result
        assert "example.com" not in result

    def test_keep_urls(self):
        """Test keeping URLs when disabled."""
        preprocessor = TextPreprocessor(remove_urls=False)
        text = "Check this: http://example.com"
        result = preprocessor.preprocess(text)
        assert "http://example.com" in result

    def test_remove_html(self):
        """Test HTML tag removal."""
        preprocessor = TextPreprocessor()
        text = "<p>Great product!</p><br/>"
        result = preprocessor.preprocess(text)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "great product!" in result

    def test_normalize_repeats(self):
        """Test repeated character normalization."""
        preprocessor = TextPreprocessor(max_repeat=3)
        text = "Soooooo goooood!!!!!!"
        result = preprocessor.preprocess(text)
        # Should limit to 3 consecutive characters
        assert "oooo" not in result
        assert "!!!!" in result  # max 3 repeats

    def test_normalize_repeats_disabled(self):
        """Test with repeat normalization disabled."""
        preprocessor = TextPreprocessor(normalize_repeats=False)
        text = "Soooooo goooood"
        result = preprocessor.preprocess(text)
        assert "oooooo" in result

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        preprocessor = TextPreprocessor()
        text = "Too     much    whitespace"
        result = preprocessor.preprocess(text)
        assert "  " not in result
        assert result == "too much whitespace"

    def test_lowercase(self):
        """Test lowercase conversion."""
        preprocessor = TextPreprocessor(lowercase=True)
        text = "LOUD NOISES"
        result = preprocessor.preprocess(text)
        assert result == "loud noises"

    def test_no_lowercase(self):
        """Test preserving case."""
        preprocessor = TextPreprocessor(lowercase=False)
        text = "LOUD NOISES"
        result = preprocessor.preprocess(text)
        assert result == "LOUD NOISES"

    def test_preprocess_empty_string(self):
        """Test preprocessing empty string."""
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess("   ") == ""

    def test_preprocess_none(self):
        """Test preprocessing None input."""
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess(None) == ""

    def test_preprocess_invalid_type(self):
        """Test preprocessing invalid type."""
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess(123) == ""
        assert preprocessor.preprocess([]) == ""

    def test_preprocess_batch(self, sample_texts):
        """Test batch preprocessing."""
        preprocessor = TextPreprocessor()
        results = preprocessor.preprocess_batch(sample_texts)
        assert len(results) == len(sample_texts)
        assert all(isinstance(r, str) for r in results)
        # Check first text is lowercased
        assert "amazing" in results[0]

    def test_preprocess_noisy_text(self, noisy_texts):
        """Test preprocessing noisy texts."""
        preprocessor = TextPreprocessor()
        results = preprocessor.preprocess_batch(noisy_texts)

        # URL removed
        assert "http://" not in results[0]
        # HTML removed
        assert "<p>" not in results[1]
        # Repeats normalized
        assert "oooo" not in results[2]
        # Whitespace normalized
        assert "  " not in results[4]

    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        preprocessor = TextPreprocessor()
        text = "This is a test!"
        features = preprocessor.extract_features(text)

        assert "length" in features
        assert "word_count" in features
        assert "avg_word_length" in features
        assert features["word_count"] == 4
        assert features["exclamation_count"] == 1

    def test_extract_features_url(self):
        """Test URL detection in features."""
        preprocessor = TextPreprocessor()
        text = "Check http://example.com"
        features = preprocessor.extract_features(text)
        assert features["has_url"] is True

    def test_extract_features_mention(self):
        """Test mention detection in features."""
        preprocessor = TextPreprocessor()
        text = "Hey @user how are you?"
        features = preprocessor.extract_features(text)
        assert features["has_mention"] is True

    def test_extract_features_hashtag(self):
        """Test hashtag detection in features."""
        preprocessor = TextPreprocessor()
        text = "This is #awesome"
        features = preprocessor.extract_features(text)
        assert features["has_hashtag"] is True

    def test_extract_features_punctuation(self):
        """Test punctuation counting."""
        preprocessor = TextPreprocessor()
        text = "What!? Really???"
        features = preprocessor.extract_features(text)
        assert features["exclamation_count"] == 1
        assert features["question_count"] == 4

    def test_extract_features_uppercase(self):
        """Test uppercase ratio."""
        preprocessor = TextPreprocessor()
        text = "LOUD"
        features = preprocessor.extract_features(text)
        assert features["uppercase_ratio"] == 1.0

    def test_is_valid_text_valid(self):
        """Test valid text validation."""
        preprocessor = TextPreprocessor()
        text = "This is a valid review text."
        assert preprocessor.is_valid_text(text) is True

    def test_is_valid_text_too_short(self):
        """Test text that is too short."""
        preprocessor = TextPreprocessor()
        assert preprocessor.is_valid_text("Short", min_length=10) is False

    def test_is_valid_text_too_long(self):
        """Test text that is too long."""
        preprocessor = TextPreprocessor()
        long_text = "a" * 6000
        assert preprocessor.is_valid_text(long_text, max_length=5000) is False

    def test_is_valid_text_empty(self):
        """Test empty text validation."""
        preprocessor = TextPreprocessor()
        assert preprocessor.is_valid_text("") is False
        assert preprocessor.is_valid_text("   ") is False

    def test_is_valid_text_none(self):
        """Test None input validation."""
        preprocessor = TextPreprocessor()
        assert preprocessor.is_valid_text(None) is False

    def test_is_valid_text_no_words(self):
        """Test text with no actual words."""
        preprocessor = TextPreprocessor()
        assert preprocessor.is_valid_text("!!! ??? ...") is False

    def test_unicode_normalization(self):
        """Test unicode character normalization."""
        preprocessor = TextPreprocessor()
        text = "Café résumé"
        result = preprocessor.preprocess(text)
        # Should normalize unicode
        assert isinstance(result, str)

    def test_clean_review_text_utility(self):
        """Test utility function."""
        text = "GREAT PRODUCT!!! http://example.com"
        result = clean_review_text(text)
        assert "http://" not in result
        assert result.islower()


class TestTextPreprocessorEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_text(self):
        """Test preprocessing very long text."""
        preprocessor = TextPreprocessor()
        text = "word " * 10000
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)

    def test_special_characters(self):
        """Test handling of special characters."""
        preprocessor = TextPreprocessor()
        text = "Test™ with® special© symbols™"
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)

    def test_mixed_languages(self):
        """Test handling of mixed languages."""
        preprocessor = TextPreprocessor()
        text = "Hello مرحبا 你好"
        result = preprocessor.preprocess(text)
        assert isinstance(result, str)

    def test_only_numbers(self):
        """Test text with only numbers."""
        preprocessor = TextPreprocessor()
        text = "123 456 789"
        result = preprocessor.preprocess(text)
        assert result == "123 456 789"

    def test_empty_after_preprocessing(self):
        """Test text that becomes empty after preprocessing."""
        preprocessor = TextPreprocessor()
        text = "http://example.com"
        result = preprocessor.preprocess(text)
        assert result == ""

    def test_multiple_preprocessing_passes(self):
        """Test that preprocessing is idempotent."""
        preprocessor = TextPreprocessor()
        text = "Test TEXT"
        result1 = preprocessor.preprocess(text)
        result2 = preprocessor.preprocess(result1)
        assert result1 == result2


class TestTextPreprocessorIntegration:
    """Integration tests for text preprocessing."""

    @pytest.mark.integration
    def test_full_pipeline_noisy_text(self):
        """Test full preprocessing pipeline on noisy text."""
        preprocessor = TextPreprocessor()
        text = """
        <p>AMAZING product!!! Check it out: http://example.com</p>
        Soooooo goooood!!! @user #awesome
        """
        result = preprocessor.preprocess(text)

        # Should not contain:
        assert "<p>" not in result
        assert "http://" not in result
        assert "oooo" not in result
        assert "  " not in result

        # Should be lowercase
        assert result.islower()

        # Should have content
        assert len(result) > 0

    @pytest.mark.integration
    def test_batch_processing_consistency(self):
        """Test that batch processing is consistent with individual."""
        preprocessor = TextPreprocessor()
        texts = ["Text 1", "Text 2", "Text 3"]

        # Process individually
        individual = [preprocessor.preprocess(t) for t in texts]

        # Process as batch
        batch = preprocessor.preprocess_batch(texts)

        assert individual == batch
