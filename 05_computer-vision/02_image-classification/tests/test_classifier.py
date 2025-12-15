"""
Tests for ImageClassifier class.

Focus: Key functionality and learning concepts.
"""

import pytest
import torch
from src.classifier import ImageClassifier


class TestImageClassifier:
    """Test image classification with transfer learning."""

    def test_model_initialization(self, device):
        """Test that model initializes correctly."""
        classifier = ImageClassifier(
            num_classes=10,
            model_name='resnet18',
            pretrained=False,  # Faster for testing
            device=device
        )

        assert classifier.num_classes == 10
        assert classifier.model_name == 'resnet18'
        assert classifier.device == torch.device(device)
        assert classifier.model is not None

    def test_model_forward_pass(self, mock_image_batch, device):
        """Test model can process a batch."""
        images, labels = mock_image_batch

        classifier = ImageClassifier(
            num_classes=10,
            pretrained=False,
            device=device
        )

        # Forward pass
        with torch.no_grad():
            outputs = classifier.model(images)

        # Check output shape
        assert outputs.shape == (8, 10)  # (batch_size, num_classes)

    def test_prediction(self, mock_image_batch, device):
        """Test single image prediction."""
        images, _ = mock_image_batch

        classifier = ImageClassifier(
            num_classes=10,
            pretrained=False,
            device=device
        )

        # Predict on single image
        probs, classes = classifier.predict(images[0], top_k=5)

        # Check output
        assert len(probs) == 5
        assert len(classes) == 5
        assert all(0 <= c < 10 for c in classes)
        assert all(0 <= p <= 1 for p in probs)
        assert probs[0] >= probs[1]  # Sorted descending

    def test_training_one_epoch(self, mock_dataloader, device):
        """Test model can train for one epoch."""
        classifier = ImageClassifier(
            num_classes=10,
            pretrained=False,
            device=device
        )

        # Train for 1 epoch
        classifier.train(
            train_loader=mock_dataloader,
            val_loader=None,
            num_epochs=1,
            learning_rate=0.001
        )

        # Check history was recorded
        assert len(classifier.history['train_loss']) == 1
        assert len(classifier.history['train_acc']) == 1

    def test_evaluation(self, mock_dataloader, device):
        """Test model evaluation."""
        classifier = ImageClassifier(
            num_classes=10,
            pretrained=False,
            device=device
        )

        # Evaluate
        metrics = classifier.evaluate(mock_dataloader)

        # Check metrics
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert 'per_class_accuracy' in metrics

        # Accuracy should be between 0 and 100
        assert 0 <= metrics['accuracy'] <= 100

        # Confusion matrix shape
        cm = metrics['confusion_matrix']
        assert cm.shape == (10, 10)

    def test_save_and_load(self, tmp_path, device):
        """Test model saving and loading."""
        # Create and train model briefly
        classifier1 = ImageClassifier(
            num_classes=10,
            pretrained=False,
            device=device
        )

        # Save
        save_path = tmp_path / "model.pth"
        classifier1.save_model(str(save_path))

        # Load into new classifier
        classifier2 = ImageClassifier(
            num_classes=10,
            pretrained=False,
            device=device
        )
        classifier2.load_model(str(save_path))

        # Models should have same weights
        for p1, p2 in zip(classifier1.model.parameters(), classifier2.model.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
