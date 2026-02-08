"""
Early Failure Prediction Metrics for PDMBench.

Provides comprehensive metrics for RUL prediction, survival analysis,
and lead time evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, roc_auc_score, f1_score,
    precision_score, recall_score, average_precision_score,
    confusion_matrix
)
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# RUL (Remaining Useful Life) Metrics
# =============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error for RUL prediction."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error for RUL prediction."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error for RUL prediction.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (0-100 scale)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return 0.0

    return float(100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))))


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray,
                          a1: float = 13.0, a2: float = 10.0) -> float:
    """
    NASA Prognostic Scoring Function (asymmetric loss).

    Penalizes late predictions more heavily than early predictions,
    as late predictions in RUL estimation are more dangerous.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        a1: Coefficient for early predictions (default: 13)
        a2: Coefficient for late predictions (default: 10)

    Returns:
        NASA score (lower is better)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    d = y_pred - y_true  # d < 0 means early prediction, d > 0 means late prediction

    score = np.where(
        d < 0,
        np.exp(-d / a1) - 1,  # Early prediction
        np.exp(d / a2) - 1    # Late prediction
    )

    return float(np.sum(score))


def normalized_nasa_score(y_true: np.ndarray, y_pred: np.ndarray,
                          a1: float = 13.0, a2: float = 10.0) -> float:
    """Normalized NASA score (per sample average)."""
    return nasa_scoring_function(y_true, y_pred, a1, a2) / len(y_true)


def early_late_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate ratio of early vs late predictions.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values

    Returns:
        Dictionary with early_ratio, late_ratio, on_time_ratio
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    d = y_pred - y_true
    n = len(y_true)

    early = np.sum(d < 0) / n
    late = np.sum(d > 0) / n
    on_time = np.sum(d == 0) / n

    return {
        'early_ratio': float(early),
        'late_ratio': float(late),
        'on_time_ratio': float(on_time),
        'mean_error': float(np.mean(d)),
        'std_error': float(np.std(d))
    }


def rul_within_tolerance(y_true: np.ndarray, y_pred: np.ndarray,
                         tolerance: float = 0.1) -> float:
    """
    Percentage of predictions within tolerance of true RUL.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        tolerance: Relative tolerance (default: 0.1 = 10%)

    Returns:
        Percentage of predictions within tolerance
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle zero RUL
    abs_tolerance = np.maximum(np.abs(y_true) * tolerance, 1.0)
    within = np.abs(y_pred - y_true) <= abs_tolerance

    return float(np.mean(within) * 100)


# =============================================================================
# Lead Time Metrics
# =============================================================================

def precision_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                          threshold: float = 0.5) -> float:
    """
    Precision at a probability threshold.

    Args:
        y_true: True binary labels (1 = failure)
        y_prob: Predicted probabilities
        threshold: Probability threshold for classification

    Returns:
        Precision score
    """
    y_pred = (y_prob >= threshold).astype(int)
    if np.sum(y_pred) == 0:
        return 0.0
    return float(precision_score(y_true, y_pred, zero_division=0))


def recall_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        threshold: float = 0.5) -> float:
    """
    Recall at a probability threshold.

    Args:
        y_true: True binary labels (1 = failure)
        y_prob: Predicted probabilities
        threshold: Probability threshold for classification

    Returns:
        Recall score
    """
    y_pred = (y_prob >= threshold).astype(int)
    return float(recall_score(y_true, y_pred, zero_division=0))


def f1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> float:
    """F1 score at a probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def precision_recall_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = [0.3, 0.5, 0.7]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, F1 at multiple thresholds.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        thresholds: List of probability thresholds

    Returns:
        Nested dictionary with metrics at each threshold
    """
    results = {}
    for t in thresholds:
        results[f'threshold_{t}'] = {
            'precision': precision_at_threshold(y_true, y_prob, t),
            'recall': recall_at_threshold(y_true, y_prob, t),
            'f1': f1_at_threshold(y_true, y_prob, t)
        }
    return results


def precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray,
                       k_percent: float = 0.1) -> float:
    """
    Precision at top-k% predictions.

    Args:
        y_true: True binary labels (1 = failure)
        y_prob: Predicted probabilities
        k_percent: Top percentage to consider (e.g., 0.1 = top 10%)

    Returns:
        Precision for top-k% predictions
    """
    n = len(y_true)
    k = max(1, int(n * k_percent))

    # Get indices of top-k predictions
    top_k_idx = np.argsort(y_prob)[-k:]

    # Calculate precision
    return float(np.mean(y_true[top_k_idx]))


def recall_at_top_k(y_true: np.ndarray, y_prob: np.ndarray,
                    k_percent: float = 0.1) -> float:
    """
    Recall at top-k% predictions.

    Args:
        y_true: True binary labels (1 = failure)
        y_prob: Predicted probabilities
        k_percent: Top percentage to consider

    Returns:
        Recall for top-k% predictions
    """
    n = len(y_true)
    k = max(1, int(n * k_percent))

    # Get indices of top-k predictions
    top_k_idx = np.argsort(y_prob)[-k:]

    # Calculate recall
    total_positive = np.sum(y_true)
    if total_positive == 0:
        return 0.0

    return float(np.sum(y_true[top_k_idx]) / total_positive)


def metrics_at_top_k_levels(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_levels: List[float] = [0.01, 0.05, 0.10, 0.20]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics at multiple top-k% levels.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        k_levels: List of k percentages

    Returns:
        Nested dictionary with metrics at each k level
    """
    results = {}
    for k in k_levels:
        results[f'top_{int(k*100)}%'] = {
            'precision': precision_at_top_k(y_true, y_prob, k),
            'recall': recall_at_top_k(y_true, y_prob, k)
        }
    return results


def average_lead_time(
    predictions: np.ndarray,
    true_failure_time: np.ndarray,
    prediction_times: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate average lead time for failure predictions.

    Lead time = time between first correct failure prediction and actual failure.

    Args:
        predictions: Predicted probabilities over time (n_samples, n_time_steps)
        true_failure_time: Actual failure time index for each sample
        prediction_times: Time indices for predictions
        threshold: Probability threshold for failure prediction

    Returns:
        Dictionary with average, median lead time and statistics
    """
    lead_times = []

    for i in range(len(predictions)):
        # Find first time prediction exceeds threshold
        failure_pred = predictions[i] >= threshold
        if np.any(failure_pred):
            first_pred_idx = np.argmax(failure_pred)
            first_pred_time = prediction_times[first_pred_idx]
            actual_failure = true_failure_time[i]

            # Lead time (positive = predicted before failure)
            lead_time = actual_failure - first_pred_time
            if lead_time >= 0:  # Only count if prediction was before failure
                lead_times.append(lead_time)

    if len(lead_times) == 0:
        return {
            'average_lead_time': 0.0,
            'median_lead_time': 0.0,
            'std_lead_time': 0.0,
            'n_correct_predictions': 0
        }

    return {
        'average_lead_time': float(np.mean(lead_times)),
        'median_lead_time': float(np.median(lead_times)),
        'std_lead_time': float(np.std(lead_times)),
        'min_lead_time': float(np.min(lead_times)),
        'max_lead_time': float(np.max(lead_times)),
        'n_correct_predictions': len(lead_times)
    }


def false_alarm_rate(y_true: np.ndarray, y_prob: np.ndarray,
                     threshold: float = 0.5) -> float:
    """
    Calculate false alarm rate (false positive rate).

    Args:
        y_true: True binary labels (1 = failure)
        y_prob: Predicted probabilities
        threshold: Probability threshold

    Returns:
        False alarm rate
    """
    y_pred = (y_prob >= threshold).astype(int)

    # False positives
    fp = np.sum((y_pred == 1) & (y_true == 0))
    # True negatives
    tn = np.sum((y_pred == 0) & (y_true == 0))

    if (fp + tn) == 0:
        return 0.0

    return float(fp / (fp + tn))


def miss_rate(y_true: np.ndarray, y_prob: np.ndarray,
              threshold: float = 0.5) -> float:
    """
    Calculate miss rate (false negative rate).

    Args:
        y_true: True binary labels (1 = failure)
        y_prob: Predicted probabilities
        threshold: Probability threshold

    Returns:
        Miss rate
    """
    y_pred = (y_prob >= threshold).astype(int)

    # False negatives
    fn = np.sum((y_pred == 0) & (y_true == 1))
    # True positives
    tp = np.sum((y_pred == 1) & (y_true == 1))

    if (fn + tp) == 0:
        return 0.0

    return float(fn / (fn + tp))


# =============================================================================
# Survival Analysis Metrics
# =============================================================================

def concordance_index(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    event_observed: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Harrell's Concordance Index (C-index).

    The C-index measures the probability that for a random pair of subjects,
    the one with the higher predicted risk score will fail first.

    Args:
        event_times: Time of event/censoring for each subject
        predicted_scores: Predicted risk scores (higher = more risk)
        event_observed: Binary indicator (1 = event occurred, 0 = censored)

    Returns:
        C-index value (0.5 = random, 1.0 = perfect)
    """
    if event_observed is None:
        event_observed = np.ones(len(event_times))

    event_times = np.array(event_times)
    predicted_scores = np.array(predicted_scores)
    event_observed = np.array(event_observed)

    n = len(event_times)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only consider comparable pairs
            if event_times[i] != event_times[j]:
                if event_times[i] < event_times[j]:
                    earlier_idx, later_idx = i, j
                else:
                    earlier_idx, later_idx = j, i

                # Only count if earlier event was observed
                if event_observed[earlier_idx] == 1:
                    if predicted_scores[earlier_idx] > predicted_scores[later_idx]:
                        concordant += 1
                    elif predicted_scores[earlier_idx] < predicted_scores[later_idx]:
                        discordant += 1
                    else:
                        tied_risk += 0.5

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5

    return float((concordant + tied_risk) / total)


def brier_score_at_time(
    event_times: np.ndarray,
    predicted_probs: np.ndarray,
    eval_time: float,
    event_observed: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Brier score at a specific time point.

    Args:
        event_times: Time of event/censoring for each subject
        predicted_probs: Predicted survival probability at eval_time
        eval_time: Time point at which to evaluate
        event_observed: Binary indicator (1 = event occurred)

    Returns:
        Brier score (lower is better)
    """
    if event_observed is None:
        event_observed = np.ones(len(event_times))

    event_times = np.array(event_times)
    predicted_probs = np.array(predicted_probs)
    event_observed = np.array(event_observed)

    n = len(event_times)
    scores = []

    for i in range(n):
        if event_times[i] <= eval_time and event_observed[i] == 1:
            # Event occurred before eval_time
            # True status = 0 (not survived)
            scores.append((predicted_probs[i] - 0) ** 2)
        elif event_times[i] > eval_time:
            # Still at risk at eval_time
            # True status = 1 (survived)
            scores.append((predicted_probs[i] - 1) ** 2)
        # Censored before eval_time: exclude from calculation

    if len(scores) == 0:
        return 0.0

    return float(np.mean(scores))


def integrated_brier_score(
    event_times: np.ndarray,
    predicted_survival_curves: np.ndarray,
    time_points: np.ndarray,
    event_observed: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Integrated Brier Score (IBS) over time.

    Args:
        event_times: Time of event/censoring for each subject
        predicted_survival_curves: Predicted survival probabilities (n_samples, n_time_points)
        time_points: Time points at which predictions are made
        event_observed: Binary indicator (1 = event occurred)

    Returns:
        Integrated Brier score
    """
    if event_observed is None:
        event_observed = np.ones(len(event_times))

    brier_scores = []
    for t_idx, t in enumerate(time_points):
        bs = brier_score_at_time(
            event_times,
            predicted_survival_curves[:, t_idx],
            t,
            event_observed
        )
        brier_scores.append(bs)

    # Integrate using trapezoidal rule
    time_range = time_points[-1] - time_points[0]
    if time_range == 0:
        return 0.0

    return float(np.trapz(brier_scores, time_points) / time_range)


# =============================================================================
# Comprehensive Evaluation Functions
# =============================================================================

def evaluate_rul_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive evaluation of RUL predictions.

    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values

    Returns:
        Dictionary with all RUL metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    early_late = early_late_ratio(y_true, y_pred)

    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'r2': float(r2_score(y_true, y_pred)),
        'nasa_score': nasa_scoring_function(y_true, y_pred),
        'normalized_nasa_score': normalized_nasa_score(y_true, y_pred),
        'within_10%_tolerance': rul_within_tolerance(y_true, y_pred, 0.10),
        'within_20%_tolerance': rul_within_tolerance(y_true, y_pred, 0.20),
        **early_late
    }


def evaluate_failure_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = [0.3, 0.5, 0.7],
    k_levels: List[float] = [0.01, 0.05, 0.10, 0.20]
) -> Dict[str, Union[float, Dict]]:
    """
    Comprehensive evaluation of failure prediction probabilities.

    Args:
        y_true: True binary labels (1 = failure)
        y_prob: Predicted probabilities
        thresholds: Probability thresholds for evaluation
        k_levels: Top-k% levels for evaluation

    Returns:
        Dictionary with all failure prediction metrics
    """
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()

    # Basic metrics
    results = {
        'auc_roc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        'average_precision': float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }

    # Metrics at thresholds
    for t in thresholds:
        prefix = f't{int(t*100)}'
        results[f'{prefix}_precision'] = precision_at_threshold(y_true, y_prob, t)
        results[f'{prefix}_recall'] = recall_at_threshold(y_true, y_prob, t)
        results[f'{prefix}_f1'] = f1_at_threshold(y_true, y_prob, t)
        results[f'{prefix}_false_alarm_rate'] = false_alarm_rate(y_true, y_prob, t)
        results[f'{prefix}_miss_rate'] = miss_rate(y_true, y_prob, t)

    # Metrics at top-k levels
    for k in k_levels:
        prefix = f'top{int(k*100)}pct'
        results[f'{prefix}_precision'] = precision_at_top_k(y_true, y_prob, k)
        results[f'{prefix}_recall'] = recall_at_top_k(y_true, y_prob, k)

    return results


def evaluate_survival_predictions(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    predicted_survival_curves: Optional[np.ndarray] = None,
    time_points: Optional[np.ndarray] = None,
    event_observed: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of survival predictions.

    Args:
        event_times: Time of event/censoring
        predicted_scores: Risk scores (higher = more risk)
        predicted_survival_curves: Survival probabilities over time
        time_points: Time points for survival curves
        event_observed: Binary indicator for events

    Returns:
        Dictionary with survival metrics
    """
    results = {
        'c_index': concordance_index(event_times, predicted_scores, event_observed)
    }

    # Add Brier scores if survival curves provided
    if predicted_survival_curves is not None and time_points is not None:
        results['integrated_brier_score'] = integrated_brier_score(
            event_times, predicted_survival_curves, time_points, event_observed
        )

        # Brier score at specific time points
        for t_idx in [len(time_points) // 4, len(time_points) // 2, 3 * len(time_points) // 4]:
            t = time_points[t_idx]
            results[f'brier_score_at_t{int(t)}'] = brier_score_at_time(
                event_times, predicted_survival_curves[:, t_idx], t, event_observed
            )

    return results
