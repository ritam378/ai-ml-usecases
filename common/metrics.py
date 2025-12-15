"""
Evaluation metrics for ML models.

This module provides comprehensive evaluation metrics for different types of ML tasks:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Regression metrics (MAE, MSE, RMSE, R²)
- Ranking metrics (NDCG, MAP, MRR)
- Custom metric tracking

Usage:
    from common.metrics import classification_metrics, MetricTracker

    metrics = classification_metrics(y_true, y_pred, y_prob)
    print(f"F1 Score: {metrics['f1_score']:.3f}")

    tracker = MetricTracker()
    tracker.log_metrics({"accuracy": 0.95, "loss": 0.05}, step=1)
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
    pos_label: int = 1
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC, optional)
        average: Averaging method for multiclass ("binary", "macro", "weighted")
        pos_label: Positive class label for binary classification

    Returns:
        Dict with metrics: accuracy, precision, recall, f1_score, etc.

    Example:
        metrics = classification_metrics(y_true, y_pred, y_prob)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )
    except ImportError:
        raise ImportError("scikit-learn is required for classification metrics")

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Handle binary vs multiclass
    if average == "binary":
        metrics["precision"] = precision_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        )
        metrics["f1_score"] = f1_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
        )
    else:
        metrics["precision"] = precision_score(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics["f1_score"] = f1_score(
            y_true, y_pred, average=average, zero_division=0
        )

    # ROC-AUC (if probabilities provided)
    if y_prob is not None:
        try:
            if average == "binary":
                # For binary classification, use probabilities of positive class
                if y_prob.ndim == 2:
                    y_prob_positive = y_prob[:, 1]
                else:
                    y_prob_positive = y_prob
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob_positive)
            else:
                # For multiclass
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_prob, average=average, multi_class="ovr"
                )
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")

    # Confusion matrix stats
    cm = confusion_matrix(y_true, y_pred)

    if average == "binary":
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        # Specificity
        if (tn + fp) > 0:
            metrics["specificity"] = tn / (tn + fp)

    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dict with metrics: mae, mse, rmse, r2, mape

    Example:
        metrics = regression_metrics(y_true, y_pred)
        print(f"RMSE: {metrics['rmse']:.3f}")
    """
    try:
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
    except ImportError:
        raise ImportError("scikit-learn is required for regression metrics")

    metrics = {}

    # Basic metrics
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["r2"] = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    mask = y_true != 0
    if mask.any():
        metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics["mape"] = float("inf")

    # Max error
    metrics["max_error"] = np.max(np.abs(y_true - y_pred))

    return metrics


def ranking_metrics(
    y_true: List[List[float]],
    y_pred: List[List[float]],
    k: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate ranking metrics for recommendation systems.

    Args:
        y_true: List of true relevance scores for each query
        y_pred: List of predicted relevance scores for each query
        k: Consider only top-k items (None for all items)

    Returns:
        Dict with metrics: ndcg, map, mrr, precision_at_k, recall_at_k

    Example:
        y_true = [[3, 2, 3, 0, 1], [2, 0, 1, 3]]
        y_pred = [[2.5, 2.0, 3.0, 0.5, 1.0], [1.5, 0.2, 1.0, 2.8]]
        metrics = ranking_metrics(y_true, y_pred, k=3)
    """
    metrics = {}

    # NDCG (Normalized Discounted Cumulative Gain)
    ndcg_scores = []
    for true_scores, pred_scores in zip(y_true, y_pred):
        ndcg_scores.append(_calculate_ndcg(true_scores, pred_scores, k))
    metrics["ndcg"] = np.mean(ndcg_scores) if ndcg_scores else 0.0

    if k:
        metrics[f"ndcg@{k}"] = metrics["ndcg"]

    # MAP (Mean Average Precision)
    ap_scores = []
    for true_scores, pred_scores in zip(y_true, y_pred):
        ap_scores.append(_calculate_average_precision(true_scores, pred_scores, k))
    metrics["map"] = np.mean(ap_scores) if ap_scores else 0.0

    # MRR (Mean Reciprocal Rank)
    rr_scores = []
    for true_scores, pred_scores in zip(y_true, y_pred):
        rr_scores.append(_calculate_reciprocal_rank(true_scores, pred_scores))
    metrics["mrr"] = np.mean(rr_scores) if rr_scores else 0.0

    # Precision and Recall at k
    if k:
        precision_scores = []
        recall_scores = []
        for true_scores, pred_scores in zip(y_true, y_pred):
            p, r = _calculate_precision_recall_at_k(true_scores, pred_scores, k)
            precision_scores.append(p)
            recall_scores.append(r)

        metrics[f"precision@{k}"] = np.mean(precision_scores) if precision_scores else 0.0
        metrics[f"recall@{k}"] = np.mean(recall_scores) if recall_scores else 0.0

    return metrics


def _calculate_ndcg(
    y_true: List[float],
    y_pred: List[float],
    k: Optional[int] = None
) -> float:
    """Calculate NDCG for a single query."""
    k = k or len(y_true)

    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1][:k]
    sorted_true = [y_true[i] for i in sorted_indices]

    # DCG
    dcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted_true))

    # Ideal DCG
    ideal_sorted = sorted(y_true, reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_sorted))

    return dcg / idcg if idcg > 0 else 0.0


def _calculate_average_precision(
    y_true: List[float],
    y_pred: List[float],
    k: Optional[int] = None
) -> float:
    """Calculate Average Precision for a single query."""
    k = k or len(y_true)

    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1][:k]

    # Binary relevance (consider scores > 0 as relevant)
    relevant_mask = np.array(y_true) > 0

    precision_sum = 0.0
    num_relevant = 0

    for i, idx in enumerate(sorted_indices):
        if relevant_mask[idx]:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precision_sum += precision_at_i

    total_relevant = relevant_mask.sum()
    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def _calculate_reciprocal_rank(
    y_true: List[float],
    y_pred: List[float]
) -> float:
    """Calculate Reciprocal Rank for a single query."""
    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1]

    # Find first relevant item
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] > 0:
            return 1.0 / (i + 1)

    return 0.0


def _calculate_precision_recall_at_k(
    y_true: List[float],
    y_pred: List[float],
    k: int
) -> tuple:
    """Calculate Precision and Recall at k for a single query."""
    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1][:k]

    # Binary relevance
    relevant_mask = np.array(y_true) > 0
    retrieved_relevant = sum(relevant_mask[i] for i in sorted_indices)

    precision = retrieved_relevant / k if k > 0 else 0.0
    total_relevant = relevant_mask.sum()
    recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

    return precision, recall


class MetricTracker:
    """
    Track and aggregate metrics across training epochs/steps.

    Usage:
        tracker = MetricTracker()
        for epoch in range(10):
            tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=epoch)

        history = tracker.get_history()
        best = tracker.get_best_metric("accuracy", mode="max")
    """

    def __init__(self):
        """Initialize metric tracker."""
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.steps: List[int] = []

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics for a step/epoch.

        Args:
            metrics: Dict of metric name to value
            step: Step number (auto-incremented if None)
        """
        if step is None:
            step = len(self.steps)

        self.steps.append(step)

        for name, value in metrics.items():
            self.history[name].append(value)

        logger.info(f"Step {step}: {metrics}")

    def get_history(self) -> Dict[str, List[float]]:
        """
        Get complete history of all metrics.

        Returns:
            Dict mapping metric names to value lists
        """
        return dict(self.history)

    def get_best_metric(
        self,
        metric_name: str,
        mode: str = "max"
    ) -> Dict[str, Any]:
        """
        Get best value for a metric.

        Args:
            metric_name: Name of metric
            mode: "max" for highest value, "min" for lowest

        Returns:
            Dict with best_value, best_step, and best_index

        Example:
            best = tracker.get_best_metric("accuracy", mode="max")
            print(f"Best accuracy: {best['best_value']:.3f} at step {best['best_step']}")
        """
        if metric_name not in self.history:
            raise ValueError(f"Metric '{metric_name}' not found in history")

        values = self.history[metric_name]

        if mode == "max":
            best_idx = np.argmax(values)
        elif mode == "min":
            best_idx = np.argmin(values)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'max' or 'min'")

        return {
            "best_value": values[best_idx],
            "best_step": self.steps[best_idx],
            "best_index": best_idx,
        }

    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get most recent metric values.

        Returns:
            Dict with latest values for all metrics
        """
        return {name: values[-1] for name, values in self.history.items()}

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self.history.clear()
        self.steps.clear()
        logger.info("Metric tracker reset")
