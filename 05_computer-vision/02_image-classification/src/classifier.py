"""
Image Classifier with Transfer Learning

This module implements transfer learning for image classification using
pre-trained ResNet models. Focus: Clear, interview-ready implementation.

Key Learning Points:
- Transfer learning: freezing/unfreezing layers
- PyTorch training loop
- Model evaluation with proper metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import time


class ImageClassifier:
    """
    Transfer learning image classifier using pre-trained ResNet.

    Simple, clear implementation for learning and interviews.
    """

    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize classifier with pre-trained model.

        Parameters:
        -----------
        num_classes : int
            Number of output classes
        model_name : str
            Pre-trained model ('resnet18', 'resnet34', 'mobilenet_v2')
        pretrained : bool
            Use pre-trained weights from ImageNet
        device : str
            'cuda', 'cpu', or None (auto-detect)
        """
        self.num_classes = num_classes
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load pre-trained model
        self.model = self._create_model(pretrained)
        self.model = self.model.to(self.device)

        # Training components (initialized in train())
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def _create_model(self, pretrained: bool = True):
        """
        Create model with transfer learning setup.

        Interview Key Point: We freeze pre-trained layers and only
        train the final classification layer initially.

        Parameters:
        -----------
        pretrained : bool
            Load ImageNet pre-trained weights

        Returns:
        --------
        model : torch.nn.Module
        """
        if self.model_name == 'resnet18':
            # Load pre-trained ResNet-18
            model = models.resnet18(pretrained=pretrained)

            # Freeze all layers initially
            # These layers have learned universal features (edges, textures)
            for param in model.parameters():
                param.requires_grad = False

            # Replace final fully connected layer
            # ResNet-18 has 512 features before the fc layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)

            # Only the new fc layer will be trained
            print(f"✓ Loaded {self.model_name} (pretrained={pretrained})")
            print(f"  Frozen: All conv layers")
            print(f"  Trainable: Final FC layer ({num_features} → {self.num_classes})")

        elif self.model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)

        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            for param in model.parameters():
                param.requires_grad = False
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return model

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Train the model using transfer learning.

        Training Strategy:
        1. Train only final layer with high learning rate
        2. Optionally fine-tune later layers with lower learning rate

        Parameters:
        -----------
        train_loader : DataLoader
            Training data
        val_loader : DataLoader
            Validation data (optional)
        num_epochs : int
            Number of training epochs
        learning_rate : float
            Initial learning rate
        weight_decay : float
            L2 regularization
        """
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")

        # Loss function
        # CrossEntropyLoss combines LogSoftmax + NLLLoss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (Adam is generally better than SGD for small datasets)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (reduce LR when plateau)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=7,  # Reduce every 7 epochs
            gamma=0.1      # Multiply by 0.1
        )

        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train one epoch
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
            else:
                val_loss, val_acc = 0.0, 0.0

            # Update learning rate
            self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) - "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        print("\n✓ Training complete!")

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
        --------
        avg_loss, accuracy : float, float
        """
        self.model.train()  # Set to training mode (enables dropout, batch norm training)

        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(train_loader, desc='Training', leave=False)

        for images, labels in pbar:
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()              # Compute gradients
            self.optimizer.step()        # Update weights

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = running_loss / total
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model on validation set.

        Returns:
        --------
        avg_loss, accuracy : float, float
        """
        self.model.eval()  # Set to evaluation mode (disables dropout, batch norm eval)

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Don't compute gradients (saves memory)
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Track metrics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set with detailed metrics.

        Parameters:
        -----------
        test_loader : DataLoader
            Test data
        class_names : list
            Class names for reporting

        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model...")

        self.model.eval()

        all_preds = []
        all_labels = []
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate metrics
        accuracy = 100. * correct / total

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Per-class accuracy
        class_acc = cm.diagonal() / cm.sum(axis=1)

        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"\nPer-Class Accuracy:")

        if class_names is not None:
            for i, acc in enumerate(class_acc):
                print(f"  {class_names[i]:12s}: {acc*100:.2f}%")
        else:
            for i, acc in enumerate(class_acc):
                print(f"  Class {i}: {acc*100:.2f}%")

        print("="*50 + "\n")

        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'per_class_accuracy': class_acc
        }

        return metrics

    def predict(
        self,
        image: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class probabilities for a single image.

        Parameters:
        -----------
        image : torch.Tensor
            Image tensor (C, H, W) or (1, C, H, W)
        top_k : int
            Return top-k predictions

        Returns:
        --------
        probs, classes : np.ndarray, np.ndarray
            Top-k probabilities and class indices
        """
        self.model.eval()

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probs = torch.softmax(output, dim=1)

            # Get top-k predictions
            top_probs, top_classes = probs.topk(top_k, dim=1)

        return top_probs.cpu().numpy()[0], top_classes.cpu().numpy()[0]

    def plot_training_history(self):
        """
        Plot training and validation metrics over epochs.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        if self.history['val_loss'][0] > 0:
            ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        if self.history['val_acc'][0] > 0:
            ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """
        Plot confusion matrix heatmap.

        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix
        class_names : list
            Class names
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        print(f"Model loaded from {filepath}")


if __name__ == '__main__':
    """
    Example usage demonstrating the full workflow.
    """
    from image_utils import load_cifar10, CIFAR10_CLASSES

    # Load CIFAR-10 data
    print("Loading CIFAR-10...")
    train_loader, test_loader = load_cifar10(
        data_dir='./data',
        batch_size=128,
        num_workers=2
    )

    # Create classifier
    print("\nCreating ResNet-18 classifier...")
    classifier = ImageClassifier(
        num_classes=10,
        model_name='resnet18',
        pretrained=True
    )

    # Train (just 2 epochs for quick test)
    print("\nTraining (2 epochs for quick test)...")
    classifier.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=2,
        learning_rate=0.001
    )

    # Evaluate
    metrics = classifier.evaluate(test_loader, class_names=CIFAR10_CLASSES)

    print("\n✓ Classifier test complete!")
    print(f"Final accuracy: {metrics['accuracy']:.2f}%")
