
"""Descriptive statistics and streaming statistics using Welford's algorithm."""

import numpy as np


# --------------------------------------------------
# Basic descriptive statistics
# --------------------------------------------------

def mean(arr, axis=None, keepdims=False):
    """Calculate the arithmetic mean of an array."""

    return np.mean(arr, axis=axis, keepdims=keepdims)


def std(arr, axis=None, ddof=0, keepdims=False):
    """Calculate the standard deviation of an array."""

    return np.std(
        arr,
        axis=axis,
        ddof=ddof,
        keepdims=keepdims
    )


def median(arr, axis=None, keepdims=False):
    """Calculate the median of an array."""

    return np.median(
        arr,
        axis=axis,
        keepdims=keepdims
    )


def minimum(arr, axis=None, keepdims=False):
    """Find the minimum value of an array."""

    return np.min(
        arr,
        axis=axis,
        keepdims=keepdims
    )


def maximum(arr, axis=None, keepdims=False):
    """Find the maximum value of an array."""

    return np.max(
        arr,
        axis=axis,
        keepdims=keepdims
    )


# --------------------------------------------------
# Histogram and quantiles
# --------------------------------------------------

def histogram(data, bins=10, bin_range=None):
    """Calculate a histogram of the input data."""

    counts, bin_edges = np.histogram(
        data,
        bins=bins,
        range=bin_range
    )

    return counts, bin_edges


def quantiles(data, q, axis=None, handle_nan=True):
    """Calculate quantiles with optional NaN handling."""

    if handle_nan:
        return np.nanquantile(data, q, axis=axis)

    return np.quantile(data, q, axis=axis)


# --------------------------------------------------
# Welford's streaming algorithm
# --------------------------------------------------

class WelfordStats:
    """Calculate running mean and variance using Welford's algorithm."""

    def __init__(self):
        self.count = 0
        self.mean_ = 0.0
        self.M2 = 0.0

    def update(self, x):
        """Add new observations to the running statistics."""

        try:
            values = np.asarray(x, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Observations must be numeric."
            ) from exc

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Observations must contain finite values."
            )

        for value in values:

            self.count += 1

            difference = value - self.mean_
            self.mean_ += difference / self.count

            difference2 = value - self.mean_

            self.M2 += difference * difference2

    @property
    def variance(self):
        """Return the sample variance of the observations."""

        if self.count < 2:
            raise ValueError(
                "At least two observations are required."
            )

        return self.M2 / (self.count - 1)


# --------------------------------------------------
# Streaming statistics
# --------------------------------------------------

class StreamingStats:
    """
    Calculate statistics as new data chunks arrive.

    Welford's algorithm is used for mean and variance.
    Values are also stored for exact quantiles and histograms.
    """

    def __init__(self):
        self._stats = []
        self._values = []
        self.n_features_in_ = None

    def update_stats(self, X_chunk):
        """Update statistics using a new data chunk."""

        try:
            X_chunk = np.asarray(X_chunk, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "X_chunk must contain numeric values."
            ) from exc

        # Convert one-dimensional input into a single-column array.
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)

        if X_chunk.ndim != 2:
            raise ValueError(
                "X_chunk must be a 1D or 2D array."
            )

        if X_chunk.shape[0] == 0 or X_chunk.shape[1] == 0:
            raise ValueError("X_chunk cannot be empty.")

        if np.any(np.isinf(X_chunk)):
            raise ValueError(
                "X_chunk cannot contain infinity."
            )

        # Set up one Welford object for each feature.
        if self.n_features_in_ is None:

            self.n_features_in_ = X_chunk.shape[1]

            for _ in range(self.n_features_in_):
                self._stats.append(WelfordStats())
                self._values.append([])

        elif X_chunk.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features must remain consistent."
            )

        # Update each feature separately.
        for i in range(self.n_features_in_):

            column = X_chunk[:, i]

            # Ignore missing values.
            valid_values = column[~np.isnan(column)]

            if len(valid_values) > 0:

                self._stats[i].update(valid_values)

                self._values[i].extend(
                    valid_values.tolist()
                )

        return self

    def mean(self):
        """Return the running mean of each feature."""

        return np.array([
            stat.mean_ if stat.count > 0 else np.nan
            for stat in self._stats
        ])

    def variance(self):
        """Return the running sample variance of each feature."""

        return np.array([
            stat.variance if stat.count >= 2 else np.nan
            for stat in self._stats
        ])

    def quantiles(self, q):
        """Calculate exact quantiles for each feature."""

        q_values = np.asarray(q, dtype=float)

        if not np.all(np.isfinite(q_values)):
            raise ValueError(
                "Quantiles must be finite."
            )

        if np.any((q_values < 0) | (q_values > 1)):
            raise ValueError(
                "Quantiles must be between 0 and 1."
            )

        results = []

        for values in self._values:

            if len(values) == 0:
                result = np.full(
                    q_values.shape,
                    np.nan
                )

            else:
                result = np.quantile(values, q)

            results.append(result)

        return np.asarray(results)

    def histogram(self, feature=0, bins=10, bin_range=None):
        """Calculate a histogram for one selected feature."""

        if self.n_features_in_ is None:
            raise ValueError(
                "No data has been processed yet."
            )

        if (
            isinstance(feature, bool)
            or not isinstance(feature, (int, np.integer))
        ):
            raise ValueError("feature must be an integer.")

        if feature < 0 or feature >= self.n_features_in_:
            raise ValueError("Invalid feature index.")

        if len(self._values[feature]) == 0:
            raise ValueError(
                "The selected feature has no valid observations."
            )

        return histogram(
            self._values[feature],
            bins=bins,
            bin_range=bin_range
        )

    def result(self):
        """Return the current statistics for all features."""

        return {
            "count": np.array([
                stat.count for stat in self._stats
            ]),
            "mean": self.mean(),
            "variance": self.variance()
        }
