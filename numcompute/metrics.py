"""
Classification and regression evaluation metrics.
"""

from __future__ import annotations
from typing import Tuple
from collections import deque
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """

    Calculate the fraction of correctly predicted labels.

    Parameters
    ----------
    y_true : np.ndarray
        Actual labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted labels, shape (n_samples,).

    Returns
    -------
    float
        Accuracy between 0 and 1.

    Raises
    ------
    ValueError
        If inputs are not 1D, have different shapes, or are empty.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    return float(np.mean(y_true == y_pred))




def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Actual binary labels (0 or 1), shape (n_samples,).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1), shape (n_samples,).

    Returns
    -------
    float
        Precision score between 0 and 1.
        Returns 0.0 when no positive predictions are made.

    Raises
    ------
    ValueError
        If arrays are empty, not 1D, have different shapes,
        or contain labels other than 0 and 1.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    if not (np.all(np.isin(y_true, [0, 1])) and
            np.all(np.isin(y_pred, [0, 1]))):
        raise ValueError("Labels must be 0 or 1.")

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return float(tp / (tp + fp))




def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Actual binary labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted binary labels, shape (n_samples,).

    Returns
    -------
    float
        Recall score between 0 and 1.
        Returns 0.0 if there are no actual positive cases.

    Raises
    ------
    ValueError
        If arrays are empty, not 1D, have different shapes,
        or contain labels other than 0 and 1.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    if not (np.all(np.isin(y_true, [0, 1])) and
            np.all(np.isin(y_pred, [0, 1]))):
        raise ValueError("Labels must be 0 or 1.")

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return float(tp / (tp + fn))




def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the F1 score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Actual binary labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted binary labels, shape (n_samples,).

    Returns
    -------
    float
        F1 score between 0 and 1.
        Returns 0.0 when precision and recall are both zero.

    Raises
    ------
    ValueError
        If inputs are empty, not 1D, have mismatched shapes,
        or contain labels other than 0 and 1.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0.0

    return float(2 * p * r / (p + r))




def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix for classification.

    Parameters
    ----------
    y_true : np.ndarray
        Actual class labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels, shape (n_samples,).

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
        Rows represent actual labels and columns represent
        predicted labels. Classes are sorted in ascending order.

    Raises
    ------
    ValueError
        If inputs are not 1D, are empty, have different
        shapes, or contain non-finite numeric labels.

    Complexity
    ----------
    Time: O(n log n + c^2)
    Space: O(n + c^2)
    where n is the number of samples and c is the
    number of distinct classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    for labels in (y_true, y_pred):
        if np.issubdtype(labels.dtype, np.inexact):
            if not np.all(np.isfinite(labels)):
                raise ValueError("Numeric labels must be finite.")

    classes = np.unique(np.concatenate((y_true, y_pred)))

    true_indices = np.searchsorted(classes, y_true)
    pred_indices = np.searchsorted(classes, y_pred)

    n_classes = len(classes)

    matrix = np.zeros((n_classes, n_classes), dtype=int)

    np.add.at(matrix, (true_indices, pred_indices), 1)

    return matrix




def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error for regression.

    Parameters
    ----------
    y_true : np.ndarray
        Actual numerical values, shape (n_samples,).
    y_pred : np.ndarray
        Predicted numerical values, shape (n_samples,).

    Returns
    -------
    float
        Mean of the squared differences.

    Raises
    ------
    ValueError
        If arrays are empty, not 1D, have different shapes,
        contain NaN or infinity, or produce overflow.
    TypeError
        If inputs are not real numeric arrays.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    for arr in (y_true, y_pred):
        if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
            raise TypeError("Inputs must contain real numeric values.")

    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Inputs must contain finite values.")

    squared_errors = np.square(y_true - y_pred)
    result = np.mean(squared_errors)

    if not np.isfinite(result):
        raise ValueError("Squared errors caused numerical overflow.")

    return float(result)




def roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ROC curve for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Actual binary labels (0 or 1), shape (n_samples,).
    y_score : np.ndarray
        Prediction scores, shape (n_samples,).
        Higher scores indicate stronger positive predictions.

    Returns
    -------
    fpr : np.ndarray
        False positive rates, shape (n_thresholds,).
    tpr : np.ndarray
        True positive rates, shape (n_thresholds,).
    thresholds : np.ndarray
        Score thresholds in descending order.
        The first threshold is infinity.

    Raises
    ------
    ValueError
        If inputs are empty, have different shapes,
        are not 1D, contain invalid values, or
        do not contain both binary classes.

    Complexity
    ----------
    Time: O(n log n) due to sorting.
    Space: O(n) for intermediate arrays.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_score.shape:
        raise ValueError("Inputs must have the same shape.")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("True labels must be 0 or 1.")

    y_score = np.asarray(y_score, dtype=float)

    if not np.all(np.isfinite(y_score)):
        raise ValueError("Scores must be finite.")

    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)

    if n_positive == 0 or n_negative == 0:
        raise ValueError("Both classes must be present.")

    # Sort scores from highest to lowest
    order = np.argsort(-y_score, kind="stable")

    sorted_scores = y_score[order]
    sorted_labels = y_true[order]

    # Calculate cumulative positive and negative counts
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)

    # Use the last position of each distinct score
    distinct_indices = np.flatnonzero(
        sorted_scores[:-1] != sorted_scores[1:]
    )

    threshold_indices = np.r_[
        distinct_indices, len(sorted_scores) - 1
    ]

    # Calculate rates
    tpr = np.r_[0.0, tp[threshold_indices] / n_positive]
    fpr = np.r_[0.0, fp[threshold_indices] / n_negative]

    # Include an initial threshold with no positives
    thresholds = np.r_[
        np.inf, sorted_scores[threshold_indices]
    ]

    return fpr, tpr, thresholds




def auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Area Under the Curve using the trapezoidal rule.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates, shape (n_points,), in ascending order.
    y : np.ndarray
        Y-coordinates, shape (n_points,).

    Returns
    -------
    float
        Area under the curve.

    Raises
    ------
    ValueError
        If inputs are not 1D, have different lengths,
        contain fewer than two points, contain non-finite
        values, or x is not in ascending order.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")

    if x.size < 2:
        raise ValueError("At least two points are required.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Inputs must contain finite values.")

    if np.any(np.diff(x) < 0):
        raise ValueError("X coordinates must be in ascending order.")

    return float(np.trapezoid(y, x))



from collections import deque



class StreamingMetrics:
    """Calculate classification metrics as data arrives in chunks."""

    def __init__(self, window_size=100):
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        self.window_size = window_size
        self.reset()

    def reset(self):
        """Clear all stored metric results."""
        self.y_true_all = []
        self.y_pred_all = []
        self.y_score_all = []

        self.scores_available = True

        # Store recent actual and predicted labels.
        self.recent_results = deque(maxlen=self.window_size)

    def update(self, y_true_chunk, y_pred_chunk, y_score=None):
        """Add the results from one new data chunk."""
        y_true_chunk = np.asarray(y_true_chunk)
        y_pred_chunk = np.asarray(y_pred_chunk)

        if y_true_chunk.ndim != 1 or y_pred_chunk.ndim != 1:
            raise ValueError("Inputs must be 1D arrays.")

        if y_true_chunk.shape != y_pred_chunk.shape:
            raise ValueError("Inputs must have the same shape.")

        if y_true_chunk.size == 0:
            raise ValueError("Input chunks cannot be empty.")

        if y_score is not None:
            y_score = np.asarray(y_score, dtype=float)

            if y_score.ndim != 1 or y_score.shape != y_true_chunk.shape:
                raise ValueError("y_score must have the same shape as y_true_chunk.")

            if not np.all(np.isfinite(y_score)):
                raise ValueError("Scores must contain finite values.")

            if not np.all(np.isin(y_true_chunk, [0, 1])):
                raise ValueError("AUC scores currently support binary labels 0 and 1.")

        # Save the complete history for cumulative metrics.
        self.y_true_all.extend(y_true_chunk.tolist())
        self.y_pred_all.extend(y_pred_chunk.tolist())

        if y_score is None:
            self.scores_available = False
        elif self.scores_available:
            self.y_score_all.extend(y_score.tolist())

        # Keep only the most recent samples for rolling metrics.
        for actual, predicted in zip(y_true_chunk, y_pred_chunk):
            self.recent_results.append((actual, predicted))

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate accuracy, precision, recall, F1 and confusion matrix."""
        matrix = confusion_matrix(y_true, y_pred)

        true_positive = np.diag(matrix).astype(float)
        predicted_total = np.sum(matrix, axis=0)
        actual_total = np.sum(matrix, axis=1)

        class_precision = np.divide(
            true_positive,
            predicted_total,
            out=np.zeros_like(true_positive),
            where=predicted_total != 0
        )

        class_recall = np.divide(
            true_positive,
            actual_total,
            out=np.zeros_like(true_positive),
            where=actual_total != 0
        )

        class_f1 = np.divide(
            2 * class_precision * class_recall,
            class_precision + class_recall,
            out=np.zeros_like(class_precision),
            where=(class_precision + class_recall) != 0
        )

        return {
            "accuracy": float(np.mean(y_true == y_pred)),
            "precision": float(np.mean(class_precision)),
            "recall": float(np.mean(class_recall)),
            "f1": float(np.mean(class_f1)),
            "confusion_matrix": matrix
        }

    def result(self):
        """Return cumulative and rolling classification metrics."""
        if not self.y_true_all:
            return {}

        # Calculate metrics using the complete history.
        y_true = np.asarray(self.y_true_all)
        y_pred = np.asarray(self.y_pred_all)

        results = self._calculate_metrics(y_true, y_pred)

        results["classes"] = np.unique(np.concatenate((y_true, y_pred)))

        # Calculate metrics using only the recent window.
        recent_data = list(self.recent_results)

        recent_true = np.asarray([item[0] for item in recent_data])
        recent_pred = np.asarray([item[1] for item in recent_data])

        rolling_results = self._calculate_metrics(
            recent_true, recent_pred
        )

        results["rolling_accuracy"] = rolling_results["accuracy"]
        results["rolling_precision"] = rolling_results["precision"]
        results["rolling_recall"] = rolling_results["recall"]
        results["rolling_f1"] = rolling_results["f1"]
        results["rolling_confusion_matrix"] = rolling_results["confusion_matrix"]

        # Calculate AUC only when scores were supplied for every update.
        if (
            self.scores_available
            and self.y_score_all
            and np.all(np.isin(y_true, [0, 1]))
            and len(np.unique(y_true)) == 2
        ):
            fpr, tpr, thresholds = roc_curve(
                y_true,
                np.asarray(self.y_score_all)
            )
            results["auc"] = auc(fpr, tpr)
        else:
            results["auc"] = None

        return results
