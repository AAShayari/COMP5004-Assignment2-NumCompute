
"""Preprocessing tools for data that arrives in chunks."""

import numpy as np


class StandardScaler:
    """Standardise numerical features using their mean and standard deviation."""

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.count_ = None
        self.n_features_in_ = None

    def partial_fit(self, X):
        """Update the running statistics with a new chunk of data."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array.")

        if np.any(np.isinf(X)):
            raise ValueError("X cannot contain infinite values.")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=int)
            self.mean_ = np.zeros(X.shape[1])
            self.var_ = np.zeros(X.shape[1])

        elif X.shape[1] != self.n_features_in_:
            raise ValueError("The number of features has changed.")

        # Calculate statistics for the current chunk.
        chunk_count = np.sum(~np.isnan(X), axis=0)
        chunk_mean = np.zeros(X.shape[1])
        chunk_var = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            values = X[:, i]
            values = values[~np.isnan(values)]

            if len(values) > 0:
                chunk_mean[i] = np.mean(values)
                chunk_var[i] = np.var(values)

        # Combine the new statistics with the previous statistics.
        old_count = self.count_.copy()
        new_count = old_count + chunk_count
        total_count = np.maximum(new_count, 1)

        difference = chunk_mean - self.mean_

        new_mean = self.mean_ + (
            difference * chunk_count / total_count
        )

        old_m2 = self.var_ * old_count
        chunk_m2 = chunk_var * chunk_count

        new_m2 = (
            old_m2
            + chunk_m2
            + difference ** 2 * old_count * chunk_count / total_count
        )

        self.mean_ = new_mean
        self.count_ = new_count

        self.var_ = np.divide(
            new_m2,
            total_count,
            out=np.zeros_like(new_m2),
            where=new_count > 0
        )

        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.scale_ == 0] = 1.0

        return self

    def fit(self, X):
        """Calculate statistics from the complete dataset."""
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.count_ = None
        self.n_features_in_ = None

        return self.partial_fit(X)

    def transform(self, X):
        """Standardise data using the learned statistics."""
        if self.mean_ is None:
            raise ValueError("The scaler has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incorrect shape.")

        if np.any(np.isinf(X)):
            raise ValueError("X cannot contain infinite values.")

        result = X.copy()

        if self.with_mean:
            result = result - self.mean_

        if self.with_std:
            result = result / self.scale_

        return result

    def fit_transform(self, X):
        """Fit the scaler and transform the data."""
        self.fit(X)
        return self.transform(X)


class Imputer:
    """Replace missing numerical values with suitable values."""

    def __init__(self, strategy="mean", fill_value=0):
        if strategy not in ("mean", "constant"):
            raise ValueError(
                "strategy must be 'mean' or 'constant'."
            )

        self.strategy = strategy
        self.fill_value = fill_value

        self.statistics_ = None
        self.sums_ = None
        self.counts_ = None
        self.n_features_in_ = None

    def partial_fit(self, X):
        """Update the imputation values using a new data chunk."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array.")

        if np.any(np.isinf(X)):
            raise ValueError("X cannot contain infinite values.")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.sums_ = np.zeros(X.shape[1])
            self.counts_ = np.zeros(X.shape[1], dtype=int)

        elif X.shape[1] != self.n_features_in_:
            raise ValueError("The number of features has changed.")

        if self.strategy == "mean":
            self.sums_ += np.nansum(X, axis=0)
            self.counts_ += np.sum(~np.isnan(X), axis=0)

            # Use fill_value if an entire feature contains missing values.
            self.statistics_ = np.divide(
                self.sums_,
                self.counts_,
                out=np.full(X.shape[1], self.fill_value, dtype=float),
                where=self.counts_ > 0
            )

        else:
            self.statistics_ = np.full(
                X.shape[1], self.fill_value, dtype=float
            )

        return self

    def fit(self, X):
        """Learn imputation values from scratch."""
        self.statistics_ = None
        self.sums_ = None
        self.counts_ = None
        self.n_features_in_ = None

        return self.partial_fit(X)

    def transform(self, X):
        """Replace missing values using the learned statistics."""
        if self.statistics_ is None:
            raise ValueError("The imputer has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incorrect shape.")

        if np.any(np.isinf(X)):
            raise ValueError("X cannot contain infinite values.")

        result = X.copy()

        for i in range(self.n_features_in_):
            result[np.isnan(result[:, i]), i] = self.statistics_[i]

        return result

    def fit_transform(self, X):
        """Fit the imputer and transform the data."""
        self.fit(X)
        return self.transform(X)


class OneHotEncoder:
    """Convert categorical features into binary numerical columns."""

    def __init__(self, handle_unknown="ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(
                "handle_unknown must be 'ignore' or 'error'."
            )

        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_in_ = None

    def partial_fit(self, X):
        """Learn categories from a new chunk of data."""
        X = np.asarray(X, dtype=object)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array.")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.categories_ = [[] for _ in range(X.shape[1])]

        elif X.shape[1] != self.n_features_in_:
            raise ValueError("The number of features has changed.")

        for i in range(self.n_features_in_):
            for value in X[:, i]:

                if value is None:
                    raise ValueError("Missing categories are not supported.")

                if isinstance(value, (float, np.floating)):
                    if np.isnan(value):
                        raise ValueError("Missing categories are not supported.")

                if value not in self.categories_[i]:
                    self.categories_[i].append(value)

        return self

    def fit(self, X):
        """Learn categories from scratch."""
        self.categories_ = None
        self.n_features_in_ = None

        return self.partial_fit(X)

    def transform(self, X):
        """Convert each category into a binary indicator column."""
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")

        X = np.asarray(X, dtype=object)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incorrect shape.")

        encoded_features = []

        for i in range(self.n_features_in_):
            categories = self.categories_[i]
            encoded = np.zeros(
                (X.shape[0], len(categories)), dtype=int
            )

            for row in range(X.shape[0]):
                value = X[row, i]

                if value not in categories:
                    if self.handle_unknown == "error":
                        raise ValueError(
                            f"Unknown category '{value}' in feature {i}."
                        )
                    continue

                category_index = categories.index(value)
                encoded[row, category_index] = 1

            encoded_features.append(encoded)

        return np.concatenate(encoded_features, axis=1)

    def fit_transform(self, X):
        """Learn categories and encode the data."""
        self.fit(X)
        return self.transform(X)
