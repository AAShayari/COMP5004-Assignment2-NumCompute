"""
Descriptive statistics, Welford streaming stats, histogram, and quantiles.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import numpy as np


def mean(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    
    """
    Compute the arithmetic mean of an array.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values. Can have any number
        of dimensions.
    axis : int or tuple of int, optional
        Axis or axes along which to calculate the mean. If None,
        calculate the mean over all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.

    Returns
    -------
    ndarray or scalar
        Mean values. If axis is specified, the corresponding
        dimensions are reduced, unless keepdims=True.

    Raises
    ------
    ValueError
        If an axis is out of bounds or the input cannot be reduced
        along the specified axis.

    Complexity
    ----------
    Time: O(N), where N is the number of input elements.
    Auxiliary space: Implementation-dependent, depending on the
    NumPy operation and output size.
    """
    return np.mean(arr, axis=axis, keepdims=keepdims)
    

def std(arr: np.ndarray, axis: Optional[int] = None, ddof: int = 0, keepdims: bool = False) -> np.ndarray:
    """
    Compute the standard deviation of array elements.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values.
    axis : int or tuple of int, optional
        Axis or axes along which to calculate standard deviation.
        If None, use all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.
    ddof : int, optional
        Delta degrees of freedom. The divisor is N - ddof,
        where N is the number of elements used in the calculation.
        Default is 0.

    Returns
    -------
    ndarray or scalar
        Standard deviation values along the specified axis or axes.

    Raises
    ------
    ValueError
        If an axis is invalid.

    Notes
    -----
    NumPy may return NaN or emit a warning when there are
    insufficient observations for the chosen ddof.

    Complexity
    ----------
    Time: O(N), where N is the number of input elements.
    Auxiliary space: Implementation-dependent, depending on the
    NumPy operation and output size.
    """
    return np.std(arr, axis=axis, keepdims=keepdims, ddof=ddof)


def median(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the median of an array.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values.
    axis : int or tuple of int, optional
        Axis or axes along which to calculate the median.
        If None, use all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.

    Returns
    -------
    ndarray or scalar
        Median values along the selected axis or axes.

    Raises
    ------
    ValueError
        If an axis is invalid or the input is empty.

    Complexity
    ----------
    Time: Typically O(N log N) as a conservative bound for
    selection or ordering operations.
    Auxiliary space: Implementation-dependent.
    """

    return np.median(arr, axis=axis, keepdims=keepdims)


def minimum(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Find the minimum value of an array.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values.
    axis : int or tuple of int, optional
        Axis or axes along which to find the minimum.
        If None, use all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.

    Returns
    -------
    ndarray or scalar
        Minimum values along the selected axis or axes.

    Raises
    ------
    ValueError
        If an axis is invalid or the input is empty.

    Complexity
    ----------
    Time: O(N), where N is the number of input elements.
    Auxiliary space: Implementation-dependent, depending on
    the NumPy operation and output size.
    """
    return np.min(arr, axis=axis, keepdims=keepdims)


def maximum(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Find the maximum value of an array.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values.
    axis : int or tuple of int, optional
        Axis or axes along which to find the maximum.
        If None, use all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.

    Returns
    -------
    ndarray or scalar
        Maximum values along the selected axis or axes.

    Raises
    ------
    ValueError
        If an axis is invalid or the input is empty.

    Complexity
    ----------
    Time: O(N), where N is the number of input elements.
    Auxiliary space: Implementation-dependent, depending on
    the NumPy operation and output size.
    """
    return np.max(arr, axis=axis, keepdims=keepdims)


def histogram(
    data: np.ndarray,
    bins: int = 10,
    bin_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Compute a histogram of the input data.

    Parameters
    ----------
    data : array_like
        Input data to distribute into histogram bins.
    bins : int or array_like, optional
        Number of equally spaced bins or explicit bin edges.
        Default is 10.
    bin_range : tuple of float, optional
        Lower and upper limits of the histogram range.
        Values outside the specified range are excluded.
        Default is None.

    Returns
    -------
    counts : ndarray
        Histogram counts, with shape (B,), where B is the number
        of bins.
    bin_edges : ndarray
        Bin boundaries, with shape (B + 1,).

    Raises
    ------
    ValueError
        If bins or the specified range are invalid.

    Complexity
    ----------
    Time: O(N + B), where N is the number of input values and
    B is the number of bins, under typical conditions.
    Auxiliary space: O(B) for the returned counts and bin edges,
    excluding implementation-dependent temporary storage.
    """
    counts, bin_edges = np.histogram(
        data, bins=bins, range=bin_range
    )

    return counts, bin_edges

def quantiles(
    data: np.ndarray,
    q: Union[float, Sequence[float]],
    axis: Optional[int] = None,
    handle_nan: bool = True,
) -> np.ndarray:
    """Compute quantiles with optional NaN handling."""

    """
    Compute quantiles of the input data.

    Parameters
    ----------
    data : array_like
        Input array containing numerical values.
    q : float or array_like of float
        Quantile or quantiles to calculate, in the interval [0, 1].
    axis : int or tuple of int, optional
        Axis or axes along which to calculate quantiles.
        If None, use all elements. Default is None.
    handle_nan : bool, optional
        If True, ignore NaN values using numpy.nanquantile.
        If False, use numpy.quantile. Default is True.

    Returns
    -------
    ndarray or scalar
        Computed quantile values. When q contains multiple
        quantiles, its dimensions precede the remaining data
        dimensions.

    Raises
    ------
    ValueError
        If q contains values outside [0, 1], an axis is invalid,
        or the selected data cannot produce a quantile.

    Complexity
    ----------
    Time: Typically O(N log N) as a conservative bound for
    ordering or selection operations.
    Auxiliary space: Implementation-dependent.
    """
    if handle_nan:
        return np.nanquantile(data, q, axis=axis)
    else:
        return np.quantile(data, q, axis=axis)

class WelfordStats:
    """Online streaming calculation of mean and variance via Welford's algorithm."""
    """
    Calculate streaming statistics using Welford's algorithm.

    Attributes
    ----------
    count : int
        Number of observations processed.
    mean_ : float
        Running arithmetic mean.
    M2 : float
        Running sum of squared deviations from the current mean.

    Notes
    -----
    The class maintains statistics incrementally without storing
    the entire history of observations.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean_ = 0.0
        self.M2 = 0.0

    def update(self, x: Union[float, np.ndarray]) -> None:
        """
        Incorporate new observations into the running statistics.

        Parameters
        ----------
        x : array_like
            One or more numerical observations. The input is
            flattened before its values are processed.

        Returns
        -------
        None
            Updates the object's internal statistics in place.

        Raises
        ------
        ValueError
            If the supplied values cannot be processed by the
            underlying numerical operations.

        Complexity
        ----------
        Time: O(K), where K is the number of new observations.
        Auxiliary space: O(K) if flattening creates a copy;
        otherwise, the additional storage depends on the input.
        """
        values = np.asarray(x).ravel()

        for value in values:
            self.count += 1

            diff1 = value - self.mean_
            self.mean_ += diff1 / self.count
            diff2 = value - self.mean_

            self.M2 += diff1 * diff2


    @property
    def variance(self) -> float:
        """
        Return the sample variance of the observations.

        Returns
        -------
        float
            Sample variance, calculated using M2 / (count - 1).

        Raises
        ------
        
        ValueError
            If fewer than two observations have been processed.

        Complexity
        ----------
        Time: O(1).
        Auxiliary space: O(1).
        """
        if self.count < 2:
            raise ValueError("Error: Atleast two observations needed")

        return self.M2 / (self.count-1)


class StreamingStats:
    """
    Calculate statistics incrementally as new data chunks arrive.

    Mean and variance are updated using Welford's algorithm.
    Observations are retained to calculate exact quantiles
    and histograms.
    """

    def __init__(self):
        self._stats = []
        self._values = []
        self.n_features_in_ = None

    def update_stats(self, X_chunk):
        """
        Update statistics using a new chunk of observations.

        Parameters
        ----------
        X_chunk : array_like
            New numerical observations. Accepts a 1D or 2D array.
            Each column represents a feature.

        Returns
        -------
        self
            Updated StreamingStats object.
        """
        X_chunk = np.asarray(X_chunk, dtype=float)

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
                "X_chunk cannot contain infinite values."
            )

        if self.n_features_in_ is None:
            self.n_features_in_ = X_chunk.shape[1]

            for _ in range(self.n_features_in_):
                self._stats.append(WelfordStats())
                self._values.append([])

        elif X_chunk.shape[1] != self.n_features_in_:
            raise ValueError(
                "The number of features must remain consistent."
            )

        for i in range(self.n_features_in_):
            column = X_chunk[:, i]

            # Ignore missing values when updating statistics.
            valid_values = column[~np.isnan(column)]

            if len(valid_values) > 0:
                self._stats[i].update(valid_values)
                self._values[i].extend(valid_values.tolist())

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
        """
        Calculate exact quantiles for each feature.

        Parameters
        ----------
        q : float or array-like
            Quantile values between 0 and 1.

        Returns
        -------
        ndarray
            Quantile values for each feature.
        """
        q_values = np.asarray(q, dtype=float)

        if not np.all(np.isfinite(q_values)) or np.any(
            (q_values < 0) | (q_values > 1)
        ):
            raise ValueError(
                "Quantiles must be between 0 and 1."
            )

        results = []

        for values in self._values:
            if len(values) == 0:
                results.append(
                    np.full(q_values.shape, np.nan)
                )
            else:
                results.append(np.quantile(values, q))

        return np.asarray(results)

    def histogram(
        self,
        feature=0,
        bins=10,
        bin_range=None
    ):
        """
        Calculate a histogram for a selected feature.

        Parameters
        ----------
        feature : int
            Column index of the feature.
        bins : int or array-like
            Number of bins or explicit bin edges.
        bin_range : tuple, optional
            Lower and upper limits of the histogram range.

        Returns
        -------
        counts, bin_edges
            Histogram counts and bin boundaries.
        """
        if self.n_features_in_ is None:
            raise ValueError(
                "No data has been processed yet."
            )

        if not isinstance(feature, (int, np.integer)):
            raise ValueError("feature must be an integer.")

        if feature < 0 or feature >= self.n_features_in_:
            raise ValueError("Invalid feature index.")

        if len(self._values[feature]) == 0:
            raise ValueError(
                "The selected feature has no valid observations."
            )

        return histogram(
            np.asarray(self._values[feature]),
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
