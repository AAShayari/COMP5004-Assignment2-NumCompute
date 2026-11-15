
"""NumPy-only decision tree classifier with buffered streaming training."""

from dataclasses import dataclass
from typing import Any, Optional
import warnings

import numpy as np


@dataclass
class _Node:
    """Represent one node in the decision tree."""

    prediction: Any
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class DecisionTreeClassifier:
    """Decision tree classifier supporting buffered incremental training.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    criterion : str
        Split criterion: "gini" or "entropy".
    min_samples_split : int
        Minimum samples required to split a node.
    max_features : None, int, float, or str
        Number of features considered at each split.
        Supported strings: "sqrt" and "log2".
    buffer_size : int
        Number of buffered samples required before retraining.
    random_state : int or None
        Seed for reproducible feature selection.
    """

    def __init__(
        self,
        max_depth: int = 5,
        criterion: str = "gini",
        min_samples_split: int = 2,
        max_features=None,
        buffer_size: int = 500,
        random_state=None,
    ):
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")

        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")

        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")

        if not isinstance(buffer_size, int) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer.")

        if max_features is not None:
            if isinstance(max_features, bool):
                raise TypeError("max_features cannot be a boolean.")
            if isinstance(max_features, int) and max_features < 1:
                raise ValueError("Integer max_features must be >= 1.")
            if isinstance(max_features, float) and not (0 < max_features <= 1):
                raise ValueError("Float max_features must be in (0, 1].")
            if isinstance(max_features, str) and max_features not in (
                "sqrt", "log2"
            ):
                raise ValueError(
                    "String max_features must be 'sqrt' or 'log2'."
                )
            if not isinstance(max_features, (int, float, str)):
                raise TypeError(
                    "max_features must be None, int, float, 'sqrt', or 'log2'."
                )

        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)

        self._root = None
        self.classes_ = None
        self.n_features_in_ = None
        self._impute_values = None

        self._buffer_X = []
        self._buffer_y = []

        self.is_fitted_ = False

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_xy(X, y):
        """Validate feature and target arrays."""

        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric feature values.") from exc

        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] == 0:
            raise ValueError("Training data cannot be empty.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if y.dtype.kind in "fc":
            if not np.all(np.isfinite(y)):
                raise ValueError("Class labels cannot contain NaN or infinity.")

        if y.dtype.kind == "O":
            if any(label is None for label in y):
                raise ValueError("Class labels cannot contain None.")

        if np.isinf(X).any():
            raise ValueError("X cannot contain positive or negative infinity.")

        return X, y

    # ---------------------------------------------------------
    # Impurity and class calculations
    # ---------------------------------------------------------

    def _impurity_from_counts(self, counts):
        """Calculate Gini impurity or entropy from class counts."""

        totals = np.sum(counts, axis=-1, keepdims=True)

        probabilities = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals != 0,
        )

        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities ** 2, axis=-1)

        positive = probabilities > 0

        terms = np.zeros_like(probabilities)
        terms[positive] = (
            probabilities[positive] * np.log2(probabilities[positive])
        )

        return -np.sum(terms, axis=-1)

    def _impurity(self, y):
        """Calculate impurity for a group of class labels."""

        _, counts = np.unique(y, return_counts=True)
        return float(self._impurity_from_counts(counts))

    @staticmethod
    def _majority(y):
        """Return the most frequent class."""

        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    # ---------------------------------------------------------
    # Feature selection
    # ---------------------------------------------------------

    def _feature_count(self, n_features):
        """Determine the number of features to consider."""

        value = self.max_features

        if value is None:
            return n_features

        if value == "sqrt":
            return max(1, int(np.sqrt(n_features)))

        if value == "log2":
            return max(1, int(np.log2(n_features)))

        if isinstance(value, float):
            return max(1, int(np.ceil(value * n_features)))

        if value > n_features:
            raise ValueError(
                "max_features cannot exceed the number of features."
            )

        return value

    # ---------------------------------------------------------
    # Best split calculation
    # ---------------------------------------------------------

    def _best_split(self, X, y):
        """Find the best feature and threshold using NumPy operations."""

        n_samples, n_features = X.shape

        feature_count = self._feature_count(n_features)

        if feature_count < n_features:
            features = np.sort(
                self._rng.choice(
                    n_features,
                    size=feature_count,
                    replace=False,
                )
            )
        else:
            features = np.arange(n_features)

        _, y_encoded = np.unique(y, return_inverse=True)
        n_classes = int(np.max(y_encoded)) + 1

        parent_counts = np.bincount(
            y_encoded, minlength=n_classes
        ).astype(float)

        parent_impurity = float(
            self._impurity_from_counts(parent_counts)
        )

        best_gain = 0.0
        best_split = None

        for feature in features:
            values = X[:, feature]

            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            sorted_labels = y_encoded[order]

            # Find positions between distinct feature values.
            split_positions = np.flatnonzero(
                sorted_values[:-1] < sorted_values[1:]
            ) + 1

            if split_positions.size == 0:
                continue

            # Encode class labels as one-hot counts.
            one_hot = np.eye(n_classes)[sorted_labels]

            cumulative_counts = np.cumsum(one_hot, axis=0)

            left_counts = cumulative_counts[split_positions - 1]
            right_counts = parent_counts - left_counts

            left_sizes = split_positions.astype(float)
            right_sizes = n_samples - left_sizes

            left_impurity = self._impurity_from_counts(left_counts)
            right_impurity = self._impurity_from_counts(right_counts)

            weighted_impurity = (
                left_sizes * left_impurity
                + right_sizes * right_impurity
            ) / n_samples

            gains = parent_impurity - weighted_impurity

            best_index = int(np.argmax(gains))
            gain = float(gains[best_index])

            if gain > best_gain + 1e-15:
                position = split_positions[best_index]

                lower = sorted_values[position - 1]
                upper = sorted_values[position]

                # Avoid overflow when calculating the midpoint.
                threshold = lower / 2.0 + upper / 2.0

                best_gain = gain
                best_split = (int(feature), float(threshold))

        return best_split

    # ---------------------------------------------------------
    # Recursive tree construction
    # ---------------------------------------------------------

    def _build(self, X, y, depth):
        """Build the decision tree recursively."""

        prediction = self._majority(y)
        node = _Node(prediction=prediction)

        if depth >= self.max_depth:
            return node

        if y.size < self.min_samples_split:
            return node

        if np.unique(y).size == 1:
            return node

        split = self._best_split(X, y)

        if split is None:
            return node

        feature, threshold = split

        left_mask = X[:, feature] <= threshold

        if not np.any(left_mask) or np.all(left_mask):
            return node

        node.feature = feature
        node.threshold = threshold

        node.left = self._build(
            X[left_mask],
            y[left_mask],
            depth + 1,
        )

        node.right = self._build(
            X[~left_mask],
            y[~left_mask],
            depth + 1,
        )

        return node

    # ---------------------------------------------------------
    # Training from scratch
    # ---------------------------------------------------------

    def fit(self, X, y):
        """Train a new decision tree from scratch."""

        X, y = self._validate_xy(X, y)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # Calculate median values for missing features.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(X, axis=0)

        self._impute_values = np.where(
            np.isnan(medians),
            0.0,
            medians,
        )

        X_clean = np.where(
            np.isnan(X),
            self._impute_values,
            X,
        )

        if not np.all(np.isfinite(X_clean)):
            raise ValueError("X contains invalid feature values.")

        self._rng = np.random.default_rng(self.random_state)

        self._root = self._build(X_clean, y, depth=0)

        self.is_fitted_ = True

        # A direct fit starts a fresh training history.
        self._buffer_X.clear()
        self._buffer_y.clear()

        return self

    # ---------------------------------------------------------
    # Incremental buffered training
    # ---------------------------------------------------------

    def partial_fit(self, X_chunk, y_chunk):
        """Add a data chunk and retrain when the buffer is full."""

        X_chunk, y_chunk = self._validate_xy(X_chunk, y_chunk)

        if (
            self.n_features_in_ is not None
            and X_chunk.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                "Incoming chunk has a different number of features."
            )

        if self._buffer_X:
            if X_chunk.shape[1] != self._buffer_X[0].shape[1]:
                raise ValueError(
                    "All buffered chunks must have the same feature count."
                )

        self._buffer_X.append(X_chunk.copy())
        self._buffer_y.append(y_chunk.copy())

        buffered = self.buffered_samples_

        if buffered >= self.buffer_size:
            X_all = np.concatenate(self._buffer_X, axis=0)
            y_all = np.concatenate(self._buffer_y, axis=0)

            self.fit(X_all, y_all)

        return self

    @property
    def buffered_samples_(self):
        """Return the number of samples waiting for retraining."""

        return sum(part.shape[0] for part in self._buffer_X)

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def _predict_one(self, row):
        """Traverse the tree to predict one sample."""

        node = self._root

        while node.feature is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction

    def predict(self, X):
        """Predict class labels for a feature matrix."""

        if not self.is_fitted_ or self._root is None:
            raise RuntimeError(
                "DecisionTreeClassifier has not been fitted yet."
            )

        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric feature values.") from exc

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features from training data."
            )

        if np.isinf(X).any():
            raise ValueError("X cannot contain positive or negative infinity.")

        X = np.where(
            np.isnan(X),
            self._impute_values,
            X,
        )

        if X.shape[0] == 0:
            return np.empty(0, dtype=self.classes_.dtype)

        predictions = [
            self._predict_one(row)
            for row in X
        ]

        return np.asarray(predictions, dtype=self.classes_.dtype)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def score(self, X, y):
        """Calculate classification accuracy."""

        predictions = self.predict(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if y.shape[0] != predictions.shape[0]:
            raise ValueError(
                "y must have one label for every prediction."
            )

        if y.size == 0:
            raise ValueError("Cannot calculate accuracy on empty data.")

        return float(np.mean(predictions == y))
