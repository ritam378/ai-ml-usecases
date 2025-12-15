"""
Tests for Data Generator Module

Tests cover:
- Review generation (single and batch)
- Sentiment distribution
- Data saving (CSV and JSON)
- Statistics generation
- Edge cases
"""

import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd

from src.data_generator import (
    SentimentDataGenerator,
    generate_sample_dataset
)


class TestSentimentDataGenerator:
    """Test cases for SentimentDataGenerator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        generator = SentimentDataGenerator()
        assert generator.seed == 42

    def test_initialization_custom_seed(self):
        """Test initialization with custom seed."""
        generator = SentimentDataGenerator(seed=123)
        assert generator.seed == 123

    def test_generate_review_positive(self):
        """Test generating positive review."""
        generator = SentimentDataGenerator()
        text, sentiment, rating = generator.generate_review("positive")

        assert isinstance(text, str)
        assert len(text) > 0
        assert sentiment == "positive"
        assert rating in [4, 5]

    def test_generate_review_negative(self):
        """Test generating negative review."""
        generator = SentimentDataGenerator()
        text, sentiment, rating = generator.generate_review("negative")

        assert isinstance(text, str)
        assert len(text) > 0
        assert sentiment == "negative"
        assert rating in [1, 2]

    def test_generate_review_neutral(self):
        """Test generating neutral review."""
        generator = SentimentDataGenerator()
        text, sentiment, rating = generator.generate_review("neutral")

        assert isinstance(text, str)
        assert len(text) > 0
        assert sentiment == "neutral"
        assert rating == 3

    def test_generate_default(self):
        """Test generating with default parameters."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=100)

        assert len(data) == 100
        assert all(isinstance(review, dict) for review in data)

    def test_generate_custom_count(self):
        """Test generating custom number of samples."""
        generator = SentimentDataGenerator()

        for count in [10, 50, 200]:
            data = generator.generate(num_samples=count)
            assert len(data) == count

    def test_generate_default_distribution(self):
        """Test default sentiment distribution (60% pos, 25% neu, 15% neg)."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=1000)

        df = pd.DataFrame(data)
        sentiment_counts = df["sentiment"].value_counts()

        # Approximate distribution (allow some variance)
        pos_ratio = sentiment_counts.get("positive", 0) / len(data)
        neu_ratio = sentiment_counts.get("neutral", 0) / len(data)
        neg_ratio = sentiment_counts.get("negative", 0) / len(data)

        assert 0.55 <= pos_ratio <= 0.65
        assert 0.20 <= neu_ratio <= 0.30
        assert 0.10 <= neg_ratio <= 0.20

    def test_generate_custom_distribution(self):
        """Test custom sentiment distribution."""
        generator = SentimentDataGenerator()
        custom_dist = {
            "positive": 0.5,
            "negative": 0.3,
            "neutral": 0.2
        }

        data = generator.generate(num_samples=1000, distribution=custom_dist)
        df = pd.DataFrame(data)
        sentiment_counts = df["sentiment"].value_counts()

        pos_ratio = sentiment_counts.get("positive", 0) / len(data)
        neg_ratio = sentiment_counts.get("negative", 0) / len(data)
        neu_ratio = sentiment_counts.get("neutral", 0) / len(data)

        assert 0.45 <= pos_ratio <= 0.55
        assert 0.25 <= neg_ratio <= 0.35
        assert 0.15 <= neu_ratio <= 0.25

    def test_generate_review_fields(self):
        """Test that generated reviews have all required fields."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=10)

        required_fields = [
            "review_id", "text", "sentiment", "rating",
            "product_id", "product_category", "verified_purchase",
            "helpful_votes", "review_date"
        ]

        for review in data:
            for field in required_fields:
                assert field in review

    def test_review_id_format(self):
        """Test review ID format."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=10)

        for i, review in enumerate(data, 1):
            # After shuffling, IDs should still be in REV format
            assert review["review_id"].startswith("REV")
            assert len(review["review_id"]) == 9  # REV + 6 digits

    def test_product_id_format(self):
        """Test product ID format."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=10)

        for review in data:
            assert review["product_id"].startswith("PROD")

    def test_product_categories(self):
        """Test product categories are valid."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=100)

        categories = [review["product_category"] for review in data]
        valid_categories = generator.PRODUCT_CATEGORIES

        assert all(cat in valid_categories for cat in categories)

    def test_verified_purchase_boolean(self):
        """Test verified_purchase is boolean."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=10)

        for review in data:
            assert isinstance(review["verified_purchase"], bool)

    def test_helpful_votes_range(self):
        """Test helpful_votes is in valid range."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=100)

        for review in data:
            assert 0 <= review["helpful_votes"] <= 50

    def test_review_date_format(self):
        """Test review date format."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=10)

        for review in data:
            # Should be YYYY-MM-DD format
            assert len(review["review_date"]) == 10
            assert review["review_date"][4] == "-"
            assert review["review_date"][7] == "-"

    def test_shuffling(self):
        """Test that reviews are shuffled."""
        generator = SentimentDataGenerator(seed=42)
        data1 = generator.generate(num_samples=100)

        generator2 = SentimentDataGenerator(seed=42)
        data2 = generator2.generate(num_samples=100)

        # Same seed should produce same order
        assert [r["review_id"] for r in data1] == [r["review_id"] for r in data2]

    def test_different_seeds_different_output(self):
        """Test different seeds produce different output."""
        gen1 = SentimentDataGenerator(seed=42)
        gen2 = SentimentDataGenerator(seed=123)

        data1 = gen1.generate(num_samples=10)
        data2 = gen2.generate(num_samples=10)

        # Different seeds should produce different text
        texts1 = [r["text"] for r in data1]
        texts2 = [r["text"] for r in data2]

        assert texts1 != texts2


class TestSentimentDataGeneratorSave:
    """Test cases for saving functionality."""

    def test_save_to_csv(self, temp_dir):
        """Test saving to CSV."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=10)

        csv_path = temp_dir / "test.csv"
        generator.save_to_csv(data, str(csv_path))

        # Verify file exists
        assert csv_path.exists()

        # Verify content
        df = pd.read_csv(csv_path)
        assert len(df) == 10
        assert "text" in df.columns
        assert "sentiment" in df.columns

    def test_save_to_json(self, temp_dir):
        """Test saving to JSON."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=10)

        json_path = temp_dir / "test.json"
        generator.save_to_json(data, str(json_path))

        # Verify file exists
        assert json_path.exists()

        # Verify content
        with open(json_path) as f:
            loaded_data = json.load(f)

        assert len(loaded_data) == 10
        assert loaded_data[0]["text"] == data[0]["text"]

    def test_csv_json_consistency(self, temp_dir):
        """Test CSV and JSON contain same data."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=20)

        csv_path = temp_dir / "test.csv"
        json_path = temp_dir / "test.json"

        generator.save_to_csv(data, str(csv_path))
        generator.save_to_json(data, str(json_path))

        # Load both
        df = pd.read_csv(csv_path)
        with open(json_path) as f:
            json_data = json.load(f)

        # Compare
        assert len(df) == len(json_data)
        assert list(df["review_id"]) == [r["review_id"] for r in json_data]


class TestSentimentDataGeneratorStatistics:
    """Test cases for statistics functionality."""

    def test_get_statistics(self):
        """Test getting statistics."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=100)

        stats = generator.get_statistics(data)

        assert "total_reviews" in stats
        assert "sentiment_distribution" in stats
        assert "rating_distribution" in stats
        assert "avg_text_length" in stats
        assert "category_distribution" in stats

    def test_statistics_total_reviews(self):
        """Test total reviews statistic."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=50)

        stats = generator.get_statistics(data)
        assert stats["total_reviews"] == 50

    def test_statistics_sentiment_distribution(self):
        """Test sentiment distribution statistic."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=100)

        stats = generator.get_statistics(data)
        sent_dist = stats["sentiment_distribution"]

        assert "positive" in sent_dist
        assert "negative" in sent_dist
        assert "neutral" in sent_dist
        assert sum(sent_dist.values()) == 100

    def test_statistics_rating_distribution(self):
        """Test rating distribution statistic."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=100)

        stats = generator.get_statistics(data)
        rating_dist = stats["rating_distribution"]

        # All ratings should be 1-5
        assert all(1 <= rating <= 5 for rating in rating_dist.keys())

    def test_statistics_avg_text_length(self):
        """Test average text length statistic."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=100)

        stats = generator.get_statistics(data)
        avg_length = stats["avg_text_length"]

        assert avg_length > 0
        assert isinstance(avg_length, float)


class TestGenerateSampleDataset:
    """Test utility function."""

    def test_generate_sample_dataset(self, temp_dir):
        """Test generate_sample_dataset utility function."""
        data = generate_sample_dataset(
            output_dir=str(temp_dir),
            num_samples=50
        )

        # Verify data returned
        assert len(data) == 50

        # Verify files created
        csv_path = temp_dir / "reviews.csv"
        json_path = temp_dir / "reviews.json"

        assert csv_path.exists()
        assert json_path.exists()

        # Verify file content
        df = pd.read_csv(csv_path)
        assert len(df) == 50

    def test_generate_sample_dataset_creates_dir(self, temp_dir):
        """Test that function creates output directory."""
        new_dir = temp_dir / "new_data"

        generate_sample_dataset(
            output_dir=str(new_dir),
            num_samples=10
        )

        assert new_dir.exists()


class TestSentimentDataGeneratorEdgeCases:
    """Test edge cases and error handling."""

    def test_generate_zero_samples(self):
        """Test generating zero samples."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=0)

        assert len(data) == 0

    def test_generate_one_sample(self):
        """Test generating single sample."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=1)

        assert len(data) == 1

    def test_distribution_normalization(self):
        """Test that distribution is normalized."""
        generator = SentimentDataGenerator()

        # Non-normalized distribution
        dist = {
            "positive": 2,
            "negative": 1,
            "neutral": 1
        }

        data = generator.generate(num_samples=100, distribution=dist)
        df = pd.DataFrame(data)

        # Should normalize to 50%, 25%, 25%
        pos_ratio = len(df[df["sentiment"] == "positive"]) / len(data)
        assert 0.45 <= pos_ratio <= 0.55

    def test_template_coverage(self):
        """Test that different templates are used."""
        generator = SentimentDataGenerator(seed=42)
        texts = set()

        for _ in range(50):
            text, _, _ = generator.generate_review("positive")
            texts.add(text)

        # Should have multiple unique texts (templates and variations)
        assert len(texts) > 1

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = SentimentDataGenerator(seed=42)
        gen2 = SentimentDataGenerator(seed=42)

        data1 = gen1.generate(num_samples=20)
        data2 = gen2.generate(num_samples=20)

        # Should be identical
        assert [r["text"] for r in data1] == [r["text"] for r in data2]
        assert [r["sentiment"] for r in data1] == [r["sentiment"] for r in data2]


class TestSentimentDataGeneratorIntegration:
    """Integration tests for data generation."""

    @pytest.mark.integration
    def test_full_pipeline(self, temp_dir):
        """Test full data generation pipeline."""
        generator = SentimentDataGenerator(seed=42)

        # Generate data
        data = generator.generate(num_samples=100)

        # Save to both formats
        csv_path = temp_dir / "reviews.csv"
        json_path = temp_dir / "reviews.json"

        generator.save_to_csv(data, str(csv_path))
        generator.save_to_json(data, str(json_path))

        # Get statistics
        stats = generator.get_statistics(data)

        # Verify everything
        assert len(data) == 100
        assert csv_path.exists()
        assert json_path.exists()
        assert stats["total_reviews"] == 100

        # Verify data quality
        df = pd.read_csv(csv_path)
        assert df["text"].notna().all()
        assert df["sentiment"].isin(["positive", "negative", "neutral"]).all()
        assert df["rating"].between(1, 5).all()

    @pytest.mark.integration
    def test_large_dataset_generation(self):
        """Test generating large dataset."""
        generator = SentimentDataGenerator()
        data = generator.generate(num_samples=10000)

        assert len(data) == 10000

        # Verify distribution
        df = pd.DataFrame(data)
        sent_dist = df["sentiment"].value_counts(normalize=True)

        # Should be close to 60/25/15
        assert 0.55 <= sent_dist.get("positive", 0) <= 0.65
