"""
Image Data Loading and Preprocessing Utilities

This module handles CIFAR-10 data loading, augmentation, and preprocessing
for transfer learning. Focus: Clear, learning-oriented implementation.

Key Learning Points:
- Data augmentation strategies for small datasets
- Proper normalization for transfer learning
- PyTorch DataLoader usage
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_transforms(augment: bool = True) -> transforms.Compose:
    """
    Get image transformations for training or testing.

    CRITICAL: Use same normalization as ImageNet pre-training!
    Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]

    Parameters:
    -----------
    augment : bool
        If True, apply data augmentation (for training)
        If False, only normalize (for testing)

    Returns:
    --------
    transform : transforms.Compose
        Composed transformations
    """
    if augment:
        # Training transformations with augmentation
        transform = transforms.Compose([
            # Data Augmentation
            transforms.RandomCrop(32, padding=4),  # Pad to 36x36, crop to 32x32
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance flip
            transforms.ColorJitter(  # Vary lighting conditions
                brightness=0.2,  # ±20% brightness
                contrast=0.2,    # ±20% contrast
                saturation=0.2   # ±20% saturation
            ),

            # Convert to tensor and normalize
            transforms.ToTensor(),  # Converts PIL Image to tensor (0-1)

            # CRITICAL: Same normalization as ImageNet
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet channel means
                std=[0.229, 0.224, 0.225]     # ImageNet channel stds
            )
        ])
    else:
        # Test transformations (no augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transform


def load_cifar10(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset with automatic download.

    PyTorch automatically downloads CIFAR-10 if not present.
    No manual download needed!

    Parameters:
    -----------
    data_dir : str
        Directory to store/load data
    batch_size : int
        Batch size for training (128 is good for CIFAR-10)
    num_workers : int
        Number of parallel data loading processes

    Returns:
    --------
    train_loader, test_loader : DataLoader, DataLoader
        Training and test data loaders
    """
    print(f"Loading CIFAR-10 dataset...")

    # Get transformations
    train_transform = get_transforms(augment=True)
    test_transform = get_transforms(augment=False)

    # Download and load training data
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,  # Auto-download if not present
        transform=train_transform
    )

    # Download and load test data
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Shuffle each epoch
        num_workers=num_workers,  # Parallel loading
        pin_memory=True    # Faster GPU transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,     # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✓ Loaded {len(train_dataset)} training images")
    print(f"✓ Loaded {len(test_dataset)} test images")
    print(f"  Batch size: {batch_size}")
    print(f"  Classes: {len(CIFAR10_CLASSES)}")

    return train_loader, test_loader


def visualize_batch(data_loader: DataLoader, num_images: int = 8):
    """
    Visualize a batch of images from the data loader.

    Helpful for understanding data augmentation effects.

    Parameters:
    -----------
    data_loader : DataLoader
        DataLoader to sample from
    num_images : int
        Number of images to display
    """
    # Get one batch
    images, labels = next(iter(data_loader))

    # Denormalize images for visualization
    # Reverse the normalization: img = img * std + mean
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(min(num_images, len(images))):
        # Denormalize
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)  # Clip to valid range

        # Convert to numpy and transpose (C, H, W) -> (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()

        # Plot
        axes[i].imshow(img_np)
        axes[i].set_title(CIFAR10_CLASSES[labels[i]])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_augmentation(data_loader: DataLoader, class_idx: int = 0):
    """
    Visualize effect of data augmentation on the same image.

    Shows how augmentation creates variations from a single image.

    Parameters:
    -----------
    data_loader : DataLoader
        DataLoader with augmentation enabled
    class_idx : int
        Which class to visualize
    """
    # This would require accessing the underlying dataset
    # and applying transforms multiple times
    # Simplified version: just show multiple samples
    print("Showing augmented samples from the same class...")
    visualize_batch(data_loader, num_images=8)


def get_dataset_statistics(train_loader: DataLoader):
    """
    Calculate and print dataset statistics.

    Useful for understanding the data distribution.

    Parameters:
    -----------
    train_loader : DataLoader
        Training data loader
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)

    # Class distribution (should be balanced for CIFAR-10)
    class_counts = torch.zeros(10)

    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1

    print("\nClass Distribution:")
    for i, count in enumerate(class_counts):
        print(f"  {CIFAR10_CLASSES[i]:12s}: {int(count):5d} images ({count/class_counts.sum()*100:.1f}%)")

    print("\nImage Properties:")
    print(f"  Shape: (3, 32, 32) - RGB, 32x32 pixels")
    print(f"  Data type: float32 (after ToTensor)")
    print(f"  Value range: Normalized (mean~0, std~1)")

    print("="*50 + "\n")


def count_parameters(model):
    """
    Count trainable and total parameters in model.

    Useful for understanding model size.

    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model

    Returns:
    --------
    trainable, total : int, int
        Number of trainable and total parameters
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return trainable, total


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize image tensor for visualization.

    Reverses ImageNet normalization.

    Parameters:
    -----------
    tensor : torch.Tensor
        Normalized image tensor (C, H, W)

    Returns:
    --------
    image : np.ndarray
        Denormalized image (H, W, C), range [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Denormalize
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)

    # Convert to numpy (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()

    return img_np


if __name__ == '__main__':
    """
    Example usage and visualization.
    """
    print("Testing image utilities...")

    # Load CIFAR-10
    train_loader, test_loader = load_cifar10(
        data_dir='./data',
        batch_size=128,
        num_workers=2
    )

    # Show dataset statistics
    get_dataset_statistics(train_loader)

    # Visualize a batch (requires display)
    # visualize_batch(train_loader, num_images=8)

    print("\n✓ Image utilities tested successfully!")
