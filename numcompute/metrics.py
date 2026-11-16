
"""Classification and regression evaluation metrics."""

from collections import deque
import numpy as np


# --------------------------------------------------
# Basic evaluation metrics
# --------------------------------------------------

def accuracy(y_true, y_pred):
    """Calculate the fraction of correctly predicted labels."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    """Calculate precision for binary classification."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if not (
        np.all(np.isin(y_true, [0, 1]))
        and np.all(np.isin(y_pred, [0, 1]))
    ):
        raise ValueError("Labels must be 0 or 1.")

    true_positive = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    false_positive = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    if true_positive + false_positive == 0:
        return 0.0

    return float(
        true_positive / (true_positive + false_positive)
    )


def recall(y_true, y_pred):
    """Calculate recall for binary classification."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if not (
        np.all(np.isin(y_true, [0, 1]))
        and np.all(np.isin(y_pred, [0, 1]))
    ):
        raise ValueError("Labels must be 0 or 1.")

    true_positive = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    false_negative = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    if true_positive + false_negative == 0:
        return 0.0

    return float(
        true_positive / (true_positive + false_negative)
    )


def f1(y_true, y_pred):
    """Calculate the F1 score for binary classification."""

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0.0

    return float(2 * p * r / (p + r))


def confusion_matrix(y_true, y_pred):
    """Calculate a confusion matrix for classification."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    classes = np.unique(
        np.concatenate((y_true, y_pred))
    )

    matrix = np.zeros(
        (len(classes), len(classes)),
        dtype=int
    )

    for actual, predicted in zip(y_true, y_pred):
        actual_index = np.where(classes == actual)[0][0]
        predicted_index = np.where(classes == predicted)[0][0]

        matrix[actual_index, predicted_index] += 1

    return matrix


def mse(y_true, y_pred):
    """Calculate Mean Squared Error for regression."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if (
        not np.issubdtype(y_true.dtype, np.number)
        or not np.issubdtype(y_pred.dtype, np.number)
        or np.iscomplexobj(y_true)
        or np.iscomplexobj(y_pred)
    ):
        raise TypeError("Inputs must contain real numeric values.")

    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    if not np.all(np.isfinite(y_true)) or not np.all(
        np.isfinite(y_pred)
    ):
        raise ValueError("Inputs must contain finite values.")

    result = np.mean((y_true - y_pred) ** 2)

    if not np.isfinite(result):
        raise ValueError("Squared errors caused numerical overflow.")

    return float(result)


# --------------------------------------------------
# ROC curve and AUC
# --------------------------------------------------

def roc_curve(y_true, y_score):
    """Calculate the ROC curve for binary classification."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if y_true.shape != y_score.shape:
        raise ValueError("Inputs must have the same shape.")

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("True labels must be 0 or 1.")

    if not np.all(np.isfinite(y_score)):
        raise ValueError("Scores must contain finite values.")

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        raise ValueError("Both classes must be present.")

    # Sort prediction scores from highest to lowest.
    order = np.argsort(-y_score, kind="stable")

    sorted_scores = y_score[order]
    sorted_labels = y_true[order]

    # Calculate cumulative true and false positives.
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)

    # Keep the last position of each distinct score.
    indices = np.flatnonzero(
        sorted_scores[:-1] != sorted_scores[1:]
    )

    indices = np.append(indices, len(sorted_scores) - 1)

    tpr = np.r_[
        0.0,
        true_positives[indices] / positives
    ]

    fpr = np.r_[
        0.0,
        false_positives[indices] / negatives
    ]

    thresholds = np.r_[
        np.inf,
        sorted_scores[indices]
    ]

    return fpr, tpr, thresholds


def auc(x, y):
    """Calculate Area Under the Curve using the trapezoidal rule."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape.")

    if len(x) < 2:
        raise ValueError("At least two points are required.")

    if not np.all(np.isfinite(x)) or not np.all(
        np.isfinite(y)
    ):
        raise ValueError("Inputs must contain finite values.")

    if np.any(np.diff(x) < 0):
        raise ValueError("X coordinates must be in ascending order.")

    # Calculate the area of each trapezoid.
    widths = np.diff(x)
    heights = (y[:-1] + y[1:]) / 2

    return float(np.sum(widths * heights))


# --------------------------------------------------
# Streaming classification metrics
# --------------------------------------------------

class StreamingMetrics:
    """Calculate cumulative and rolling metrics as data arrives."""

    def __init__(self, window_size=100):

        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, (int, np.integer))
            or window_size <= 0
        ):
            raise ValueError(
                "window_size must be a positive integer."
            )

        self.window_size = int(window_size)

        self.reset()

    def reset(self):
        """Clear all stored metric results."""

        self.y_true_all = []
        self.y_pred_all = []
        self.y_score_all = []

        self.scores_available = True

        # Keep only the most recent observations.
        self.recent_results = deque(
            maxlen=self.window_size
        )

    def update(self, y_true_chunk, y_pred_chunk, y_score=None):
        """Add the results from a new data chunk."""

        y_true_chunk = np.asarray(y_true_chunk)
        y_pred_chunk = np.asarray(y_pred_chunk)

        if y_true_chunk.ndim != 1 or y_pred_chunk.ndim != 1:
            raise ValueError("Inputs must be 1D arrays.")

        if y_true_chunk.shape != y_pred_chunk.shape:
            raise ValueError("Inputs must have the same shape.")

        if len(y_true_chunk) == 0:
            raise ValueError("Input chunks cannot be empty.")

        if y_score is not None:

            y_score = np.asarray(y_score, dtype=float)

            if y_score.ndim != 1:
                raise ValueError("y_score must be a 1D array.")

            if y_score.shape != y_true_chunk.shape:
                raise ValueError(
                    "y_score must have the same shape as y_true_chunk."
                )

            if not np.all(np.isfinite(y_score)):
                raise ValueError("Scores must contain finite values.")

            if not np.all(np.isin(y_true_chunk, [0, 1])):
                raise ValueError(
                    "AUC currently supports binary labels 0 and 1."
                )

        # Store observations for cumulative metrics.
        self.y_true_all.extend(y_true_chunk.tolist())
        self.y_pred_all.extend(y_pred_chunk.tolist())

        # AUC is available only if every update includes scores.
        if y_score is None:
            self.scores_available = False

        elif self.scores_available:
            self.y_score_all.extend(y_score.tolist())

        # Store recent observations for rolling metrics.
        for actual, predicted in zip(
            y_true_chunk, y_pred_chunk
        ):
            self.recent_results.append((actual, predicted))

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate classification metrics for given observations."""

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

        if len(self.y_true_all) == 0:
            return {}

        # Calculate metrics using all observations.
        y_true = np.asarray(self.y_true_all)
        y_pred = np.asarray(self.y_pred_all)

        results = self._calculate_metrics(
            y_true, y_pred
        )

        results["classes"] = np.unique(
            np.concatenate((y_true, y_pred))
        )

        # Calculate metrics using the recent window.
        recent_data = list(self.recent_results)

        recent_true = np.asarray([
            item[0] for item in recent_data
        ])

        recent_pred = np.asarray([
            item[1] for item in recent_data
        ])

        rolling = self._calculate_metrics(
            recent_true, recent_pred
        )

        results["rolling_accuracy"] = rolling["accuracy"]
        results["rolling_precision"] = rolling["precision"]
        results["rolling_recall"] = rolling["recall"]
        results["rolling_f1"] = rolling["f1"]

        results["rolling_confusion_matrix"] = (
            rolling["confusion_matrix"]
        )

        # Calculate AUC when scores are available for all chunks.
        if (
            self.scores_available
            and len(self.y_score_all) == len(y_true)
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
