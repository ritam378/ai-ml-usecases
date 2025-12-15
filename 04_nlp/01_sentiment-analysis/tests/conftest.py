"""
Pytest Configuration and Fixtures for Sentiment Analysis Tests

This module provides shared fixtures and configuration for all tests.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, List

import torch


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_texts() -> List[str]:
    """Provide sample texts for testing."""
    return [
        "This product is amazing! Love it!",
        "Terrible quality. Very disappointed.",
        "It's okay. Nothing special.",
        "Best purchase ever! Highly recommend!!!",
        "Do not buy. Waste of money.",
        "Works as described. Fair price.",
    ]


@pytest.fixture
def sample_reviews() -> List[Dict]:
    """Provide sample review data."""
    return [
        {
            "review_id": "REV001",
            "text": "Excellent product! Highly recommend.",
            "sentiment": "positive",
            "rating": 5,
        },
        {
            "review_id": "REV002",
            "text": "Poor quality. Not worth it.",
            "sentiment": "negative",
            "rating": 2,
        },
        {
            "review_id": "REV003",
            "text": "It's okay. Does the job.",
            "sentiment": "neutral",
            "rating": 3,
        },
    ]


@pytest.fixture
def noisy_texts() -> List[str]:
    """Provide noisy texts that need preprocessing."""
    return [
        "Check this out: http://example.com AMAZING!!!",
        "<p>Great product!</p>",
        "Soooooo goooood!!! 😍",
        "@user This is #awesome",
        "   Too     much    whitespace   ",
        "LOUD NOISES!!!",
    ]


@pytest.fixture
def invalid_texts() -> List:
    """Provide invalid text inputs."""
    return [
        "",
        None,
        123,
        [],
        {"text": "invalid"},
        "   ",
        "abc",  # too short
    ]


@pytest.fixture
def mock_model_output():
    """Mock model output for testing."""
    class MockOutput:
        def __init__(self):
            self.logits = torch.tensor([[0.2, 0.8]])  # Positive prediction

    return MockOutput()


@pytest.fixture
def sentiment_labels():
    """Standard sentiment labels."""
    return ["negative", "neutral", "positive"]


@pytest.fixture(scope="session")
def device():
    """Get device for testing (prefer CPU for tests)."""
    return "cpu"


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "model: marks tests that require model loading"
    )
