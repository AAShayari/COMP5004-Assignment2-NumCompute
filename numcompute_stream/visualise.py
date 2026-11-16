
"""Visualisation functions for streaming machine learning."""

import numpy as np
import matplotlib.pyplot as plt


def _prepare_values(values, name):
    """Check that the input contains valid one-dimensional numeric data."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if len(values) == 0:
        raise ValueError(f"{name} cannot be empty.")

    if np.any(np.isinf(values)):
        raise ValueError(f"{name} cannot contain infinite values.")

    return values


def _finish_plot(fig, save_path=None, show=True):
    """Save or display a plot and return its figure and axes."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, fig.axes[0]


def plot_metric_over_time(
    metric_values,
    title="Metric over Time",
    ylabel="Metric Value",
    save_path=None,
    show=True
):
    """
    Plot how a metric changes across streaming chunks.

    Parameters
    ----------
    metric_values : array-like
        Metric values recorded after each chunk.
    title : str
        Title of the plot.
    ylabel : str
        Label for the y-axis.
    save_path : str, optional
        File path for saving the plot.
    show : bool
        Whether to display the plot.
    """
    values = _prepare_values(metric_values, "metric_values")

    fig, ax = plt.subplots(figsize=(8, 5))

    chunks = np.arange(1, len(values) + 1)

    ax.plot(chunks, values, marker="o", label=ylabel)
    ax.set_title(title)
    ax.set_xlabel("Chunk Number")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return _finish_plot(fig, save_path, show)


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    save_path=None,
    show=True
):
    """
    Compare the performance of two models across streaming chunks.

    Parameters
    ----------
    metric1 : array-like
        Metric values for the first model.
    metric2 : array-like
        Metric values for the second model.
    labels : tuple or list
        Names of the two models.
    save_path : str, optional
        File path for saving the plot.
    show : bool
        Whether to display the plot.
    """
    values1 = _prepare_values(metric1, "metric1")
    values2 = _prepare_values(metric2, "metric2")

    if len(values1) != len(values2):
        raise ValueError("Both models must have the same number of metric values.")

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two model names.")

    fig, ax = plt.subplots(figsize=(8, 5))

    chunks = np.arange(1, len(values1) + 1)

    ax.plot(chunks, values1, marker="o", label=labels[0])
    ax.plot(chunks, values2, marker="s", label=labels[1])

    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Chunk Number")
    ax.set_ylabel("Metric Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return _finish_plot(fig, save_path, show)


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    save_path=None,
    show=True
):
    """
    Compare actual and predicted values.

    Parameters
    ----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.
    save_path : str, optional
        File path for saving the plot.
    show : bool
        Whether to display the plot.
    """
    actual = _prepare_values(y_true, "y_true")
    predicted = _prepare_values(y_pred, "y_pred")

    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length.")

    positions = np.arange(1, len(actual) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(positions, actual, marker="o", label="Ground Truth")
    ax.plot(positions, predicted, marker="x", label="Predictions")

    ax.set_title("Predictions vs Ground Truth")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Target Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return _finish_plot(fig, save_path, show)
