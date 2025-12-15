"""
Pytest fixtures for image classification tests.

Provides reusable test data and mock components.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader


@pytest.fixture
def mock_image_batch():
    """
    Create mock batch of images.

    Returns batch of 8 images (3x32x32) with labels.
    """
    # Create random images: (batch_size, channels, height, width)
    images = torch.randn(8, 3, 32, 32)

    # Create random labels (0-9 for CIFAR-10)
    labels = torch.randint(0, 10, (8,))

    return images, labels


@pytest.fixture
def mock_dataloader():
    """
    Create mock DataLoader for testing.

    Returns DataLoader with 32 samples, 10 classes.
    """
    # Create small synthetic dataset
    num_samples = 32
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    return dataloader


@pytest.fixture
def device():
    """Get device for testing (CPU for consistency)."""
    return 'cpu'
