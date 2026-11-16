
"""Decision tree classifier with buffered streaming training."""

import numpy as np


class _Node:
    """Store information about one node in the tree."""

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None


class DecisionTreeClassifier:
    """A simple decision tree classifier with buffered retraining."""

    def __init__(
        self,
        max_depth=5,
        criterion="gini",
        min_samples_split=2,
        max_features=None,
        buffer_size=500,
        random_state=None
    ):
        # Validate the configuration.
        if isinstance(max_depth, bool) or not isinstance(
            max_depth, (int, np.integer)
        ) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")

        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")

        if isinstance(min_samples_split, bool) or not isinstance(
            min_samples_split, (int, np.integer)
        ) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")

        if isinstance(buffer_size, bool) or not isinstance(
            buffer_size, (int, np.integer)
        ) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer.")

        if isinstance(max_features, bool):
            raise ValueError("Invalid max_features value.")

        if max_features is not None:
            if isinstance(max_features, (int, np.integer)):
                if max_features < 1:
                    raise ValueError("max_features must be at least 1.")

            elif isinstance(max_features, (float, np.floating)):
                if not 0 < max_features <= 1:
                    raise ValueError(
                        "Float max_features must be in (0, 1]."
                    )

            elif isinstance(max_features, str):
                if max_features not in ("sqrt", "log2"):
                    raise ValueError(
                        "Use 'sqrt' or 'log2' for max_features."
                    )

            else:
                raise TypeError("Invalid max_features value.")

        self.max_depth = int(max_depth)
        self.criterion = criterion
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

        self.root = None
        self.classes_ = None
        self.n_features_in_ = None
        self.impute_values_ = None
        self.is_fitted_ = False

        # Store incoming chunks until retraining is required.
        self.buffer_X = []
        self.buffer_y = []

    # --------------------------------------------------
    # Impurity calculation
    # --------------------------------------------------

    def _impurity(self, y):
        """Calculate Gini impurity or entropy."""

        if len(y) == 0:
            return 0.0

        labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities ** 2)

        return -np.sum(
            probabilities * np.log2(probabilities)
        )

    def _leaf_value(self, y):
        """Return the most common class label."""

        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    # --------------------------------------------------
    # Feature selection
    # --------------------------------------------------

    def _get_features(self, n_features):
        """Select the features to consider at a split."""

        if self.max_features is None:
            count = n_features

        elif self.max_features == "sqrt":
            count = max(1, int(np.sqrt(n_features)))

        elif self.max_features == "log2":
            count = max(1, int(np.log2(n_features)))

        elif isinstance(self.max_features, (float, np.floating)):
            count = max(
                1,
                int(np.ceil(self.max_features * n_features))
            )

        else:
            count = int(self.max_features)

        if count > n_features:
            raise ValueError(
                "max_features cannot exceed the number of features."
            )

        if count == n_features:
            return np.arange(n_features)

        return np.sort(
            self.rng.choice(n_features, count, replace=False)
        )

    # --------------------------------------------------
    # Finding the best split
    # --------------------------------------------------

    def _best_split(self, X, y):
        """Find the feature and threshold with the lowest impurity."""

        best_feature = None
        best_threshold = None
        best_score = self._impurity(y)

        features = self._get_features(X.shape[1])

        for feature in features:

            values = np.unique(X[:, feature])

            if len(values) < 2:
                continue

            # Try the midpoint between every pair of unique values.
            thresholds = (
                values[:-1] / 2.0 + values[1:] / 2.0
            )

            for threshold in thresholds:

                mask = X[:, feature] <= threshold

                left_y = y[mask]
                right_y = y[~mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                # Calculate weighted impurity.
                score = (
                    len(left_y) * self._impurity(left_y)
                    + len(right_y) * self._impurity(right_y)
                ) / len(y)

                if score < best_score - 1e-12:
                    best_feature = feature
                    best_threshold = threshold
                    best_score = score

        return best_feature, best_threshold

    # --------------------------------------------------
    # Building the tree
    # --------------------------------------------------

    def _build(self, X, y, depth):
        """Build the decision tree recursively."""

        node = _Node()

        # Check whether this node should become a leaf.
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            node.value = self._leaf_value(y)
            return node

        feature, threshold = self._best_split(X, y)

        # No useful split was found.
        if feature is None:
            node.value = self._leaf_value(y)
            return node

        mask = X[:, feature] <= threshold

        node.feature = feature
        node.threshold = threshold

        # Build the left and right branches.
        node.left = self._build(
            X[mask], y[mask], depth + 1
        )

        node.right = self._build(
            X[~mask], y[~mask], depth + 1
        )

        return node

    # --------------------------------------------------
    # Preparing and validating input data
    # --------------------------------------------------

    def _validate_data(self, X, y):
        """Check the input feature and target arrays."""

        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric values.") from exc

        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        if np.isinf(X).any():
            raise ValueError("X cannot contain infinity.")

        if y.dtype.kind in "fc":
            if not np.all(np.isfinite(y)):
                raise ValueError(
                    "Class labels cannot contain NaN or infinity."
                )

        if y.dtype.kind == "O":
            for label in y:
                if label is None:
                    raise ValueError(
                        "Class labels cannot contain None."
                    )

                if isinstance(label, (float, np.floating)):
                    if not np.isfinite(label):
                        raise ValueError(
                            "Class labels cannot contain NaN or infinity."
                        )

        # Ensure class labels can be compared and sorted.
        try:
            np.unique(y)
        except TypeError as exc:
            raise ValueError(
                "Class labels must have compatible data types."
            ) from exc

        return X, y

    def _prepare_features(self, X, calculate_medians=False):
        """Replace missing feature values with their median."""

        if calculate_medians:
            medians = []

            for column in range(X.shape[1]):
                values = X[:, column]
                valid_values = values[~np.isnan(values)]

                if len(valid_values) == 0:
                    medians.append(0.0)
                else:
                    medians.append(np.median(valid_values))

            self.impute_values_ = np.array(medians)

        return np.where(
            np.isnan(X),
            self.impute_values_,
            X
        )

    # --------------------------------------------------
    # Training from scratch
    # --------------------------------------------------

    def fit(self, X, y):
        """Train a new decision tree from scratch."""

        X, y = self._validate_data(X, y)

        if (
            isinstance(self.max_features, (int, np.integer))
            and self.max_features > X.shape[1]
        ):
            raise ValueError(
                "max_features cannot exceed the number of features."
            )

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        X = self._prepare_features(X, calculate_medians=True)

        # Reset the random generator for reproducible training.
        self.rng = np.random.default_rng(self.random_state)

        self.root = self._build(X, y, depth=0)

        self.is_fitted_ = True

        # Clear previously stored chunks after training.
        self.buffer_X = []
        self.buffer_y = []

        return self

    # --------------------------------------------------
    # Incremental training
    # --------------------------------------------------

    def partial_fit(self, X_chunk, y_chunk):
        """Store a chunk and retrain when the buffer is full."""

        X_chunk, y_chunk = self._validate_data(
            X_chunk, y_chunk
        )

        if self.n_features_in_ is not None:
            if X_chunk.shape[1] != self.n_features_in_:
                raise ValueError(
                    "The number of features has changed."
                )

        if self.buffer_X:
            if X_chunk.shape[1] != self.buffer_X[0].shape[1]:
                raise ValueError(
                    "Buffered chunks must have the same number of features."
                )

        self.buffer_X.append(X_chunk.copy())
        self.buffer_y.append(y_chunk.copy())

        buffered_samples = self.buffered_samples_

        if buffered_samples >= self.buffer_size:

            X_all = np.concatenate(self.buffer_X, axis=0)
            y_all = np.concatenate(self.buffer_y, axis=0)

            # Retrain using all buffered samples.
            self.fit(X_all, y_all)

        return self

    @property
    def buffered_samples_(self):
        """Return the number of samples waiting for retraining."""

        return sum(len(chunk) for chunk in self.buffer_X)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def _predict_one(self, row):
        """Follow the tree branches to predict one sample."""

        node = self.root

        while node.value is None:

            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def predict(self, X):
        """Predict class labels for new samples."""

        if not self.is_fitted_:
            raise RuntimeError(
                "The model has not been trained yet."
            )

        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric values.") from exc

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features does not match training."
            )

        if np.isinf(X).any():
            raise ValueError("X cannot contain infinity.")

        X = self._prepare_features(X)

        if len(X) == 0:
            return np.empty(0, dtype=self.classes_.dtype)

        predictions = []

        for row in X:
            predictions.append(self._predict_one(row))

        return np.asarray(
            predictions,
            dtype=self.classes_.dtype
        )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    def score(self, X, y):
        """Calculate classification accuracy."""

        predictions = self.predict(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if len(y) != len(predictions):
            raise ValueError(
                "X and y must have the same number of samples."
            )

        if len(y) == 0:
            raise ValueError(
                "Cannot calculate accuracy on empty data."
            )

        return float(np.mean(predictions == y))
