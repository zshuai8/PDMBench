"""
PDM Evaluators Module

Comprehensive evaluation metrics and analysis for PDM models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray
    probabilities: Optional[np.ndarray] = None
    auc_roc: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'per_class_precision': self.per_class_precision.tolist(),
            'per_class_recall': self.per_class_recall.tolist(),
            'per_class_f1': self.per_class_f1.tolist(),
            'auc_roc': self.auc_roc
        }

    def summary(self) -> str:
        """Return a string summary of the metrics."""
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"Accuracy:  {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1 Score:  {self.f1:.4f}",
        ]
        if self.auc_roc is not None:
            lines.append(f"AUC-ROC:   {self.auc_roc:.4f}")

        lines.extend([
            "",
            "Per-class metrics:",
            "-" * 50
        ])

        for i, (p, r, f) in enumerate(zip(
            self.per_class_precision,
            self.per_class_recall,
            self.per_class_f1
        )):
            lines.append(f"  Class {i}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class Evaluator:
    """
    Comprehensive evaluator for PDM models.

    Example:
        evaluator = Evaluator(model, test_loader)
        metrics = evaluator.evaluate()
        print(metrics)

        # Get detailed analysis
        analysis = evaluator.error_analysis()
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: Optional[torch.device] = None,
        num_classes: Optional[int] = None
    ):
        self.model = model
        self.data_loader = data_loader

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.num_classes = num_classes

        # Store predictions after evaluation
        self._predictions = None
        self._true_labels = None
        self._probabilities = None

    @torch.no_grad()
    def evaluate(self) -> EvaluationMetrics:
        """
        Run full evaluation on the data loader.

        Returns:
            EvaluationMetrics object with all computed metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        for x, y in self.data_loader:
            x = x.to(self.device)

            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probabilities.append(probabilities.cpu().numpy())

        self._predictions = np.array(all_predictions)
        self._true_labels = np.array(all_labels)
        self._probabilities = np.vstack(all_probabilities)

        # Compute metrics
        accuracy = accuracy_score(self._true_labels, self._predictions)
        precision = precision_score(
            self._true_labels, self._predictions, average='weighted', zero_division=0
        )
        recall = recall_score(
            self._true_labels, self._predictions, average='weighted', zero_division=0
        )
        f1 = f1_score(
            self._true_labels, self._predictions, average='weighted', zero_division=0
        )

        # Per-class metrics
        per_class_precision = precision_score(
            self._true_labels, self._predictions, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            self._true_labels, self._predictions, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            self._true_labels, self._predictions, average=None, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(self._true_labels, self._predictions)

        # AUC-ROC (if applicable)
        auc = None
        num_classes = len(np.unique(self._true_labels))
        if num_classes == 2:
            auc = roc_auc_score(self._true_labels, self._probabilities[:, 1])
        elif num_classes > 2:
            try:
                auc = roc_auc_score(
                    self._true_labels, self._probabilities,
                    multi_class='ovr', average='weighted'
                )
            except ValueError:
                auc = None

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            predictions=self._predictions,
            true_labels=self._true_labels,
            probabilities=self._probabilities,
            auc_roc=auc
        )

    def error_analysis(self) -> Dict[str, Any]:
        """
        Perform detailed error analysis.

        Returns:
            Dictionary with error analysis results
        """
        if self._predictions is None:
            self.evaluate()

        # Find misclassified samples
        misclassified_mask = self._predictions != self._true_labels
        misclassified_indices = np.where(misclassified_mask)[0]

        # Analyze confusion patterns
        cm = confusion_matrix(self._true_labels, self._predictions)
        confusion_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': i,
                        'predicted_class': j,
                        'count': int(cm[i, j])
                    })

        # Sort by confusion count
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

        # Confidence analysis
        correct_mask = ~misclassified_mask
        correct_confidences = self._probabilities[correct_mask].max(axis=1)
        incorrect_confidences = self._probabilities[misclassified_mask].max(axis=1)

        return {
            'num_misclassified': len(misclassified_indices),
            'misclassified_indices': misclassified_indices.tolist(),
            'error_rate': len(misclassified_indices) / len(self._true_labels),
            'top_confusions': confusion_pairs[:10],
            'avg_correct_confidence': float(correct_confidences.mean()) if len(correct_confidences) > 0 else 0,
            'avg_incorrect_confidence': float(incorrect_confidences.mean()) if len(incorrect_confidences) > 0 else 0,
        }

    def get_worst_predictions(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the n worst predictions (highest confidence wrong predictions).

        Returns:
            List of dictionaries with sample info
        """
        if self._predictions is None:
            self.evaluate()

        misclassified_mask = self._predictions != self._true_labels
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            return []

        # Get confidence of wrong predictions
        wrong_confidences = self._probabilities[misclassified_indices].max(axis=1)

        # Sort by confidence (highest first - most confident wrong predictions)
        sorted_indices = np.argsort(wrong_confidences)[::-1][:n]

        worst = []
        for idx in sorted_indices:
            sample_idx = misclassified_indices[idx]
            worst.append({
                'sample_index': int(sample_idx),
                'true_label': int(self._true_labels[sample_idx]),
                'predicted_label': int(self._predictions[sample_idx]),
                'confidence': float(wrong_confidences[idx]),
                'probabilities': self._probabilities[sample_idx].tolist()
            })

        return worst

    def class_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get per-class performance summary.

        Returns:
            Dictionary mapping class id to metrics
        """
        if self._predictions is None:
            self.evaluate()

        classes = np.unique(self._true_labels)
        summary = {}

        for cls in classes:
            cls_mask = self._true_labels == cls
            cls_predictions = self._predictions[cls_mask]
            cls_correct = (cls_predictions == cls).sum()

            summary[int(cls)] = {
                'total_samples': int(cls_mask.sum()),
                'correct_predictions': int(cls_correct),
                'accuracy': float(cls_correct / cls_mask.sum()),
                'precision': float(precision_score(
                    self._true_labels == cls, self._predictions == cls, zero_division=0
                )),
                'recall': float(recall_score(
                    self._true_labels == cls, self._predictions == cls, zero_division=0
                )),
                'f1': float(f1_score(
                    self._true_labels == cls, self._predictions == cls, zero_division=0
                ))
            }

        return summary


def compare_models(
    models: Dict[str, nn.Module],
    data_loader: DataLoader,
    device: Optional[torch.device] = None
) -> Dict[str, EvaluationMetrics]:
    """
    Compare multiple models on the same data.

    Args:
        models: Dictionary mapping model names to model instances
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary mapping model names to their evaluation metrics
    """
    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        evaluator = Evaluator(model, data_loader, device)
        metrics = evaluator.evaluate()
        results[name] = metrics

    return results


def print_comparison_table(results: Dict[str, EvaluationMetrics]):
    """Print a comparison table of model results."""
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)

    # Sort by accuracy
    sorted_models = sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True)

    for name, metrics in sorted_models:
        print(
            f"{name:<20} "
            f"{metrics.accuracy:<12.4f} "
            f"{metrics.precision:<12.4f} "
            f"{metrics.recall:<12.4f} "
            f"{metrics.f1:<12.4f}"
        )

    print("=" * 70)
