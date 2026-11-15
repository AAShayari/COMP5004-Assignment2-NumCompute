
"""Tools for training and evaluating models with streaming data."""

import sys
import numpy as np


class StreamTrainer:
    """Train a model or pipeline using incoming data chunks."""

    def __init__(self, model=None, pipeline=None, metrics=None):
        if model is not None and pipeline is not None:
            raise ValueError("Provide either a model or a pipeline, not both.")

        if model is None and pipeline is None:
            raise ValueError("A model or pipeline must be provided.")

        # The trainer can work with a model directly or a pipeline.
        self.model = model if model is not None else pipeline
        self.pipeline = pipeline
        self.metrics = metrics

        # Store the results from each processed chunk.
        self.chunk_metrics = []

        # Keep track of overall prediction accuracy.
        self.total_correct = 0
        self.total_samples = 0

        # Keep track of the most recent input chunk.
        self.last_chunk_memory = 0

    def _check_data(self, X, y):
        """Check that the input data has the correct dimensions."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] == 0:
            raise ValueError("The data chunk cannot be empty.")

        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows.")

        return X, y

    def fit_chunk(self, X_chunk, y_chunk):
        """Train the model or pipeline using one incoming chunk."""
        X_chunk, y_chunk = self._check_data(X_chunk, y_chunk)

        if not hasattr(self.model, "partial_fit"):
            raise TypeError("The model or pipeline must support partial_fit().")

        # Update the model or pipeline with the new data.
        self.model.partial_fit(X_chunk, y_chunk)

        self.last_chunk_memory = (
            X_chunk.nbytes + y_chunk.nbytes
        )

        return self

    def score_chunk(self, X_chunk, y_chunk):
        """Evaluate the model and record the results for one chunk."""
        X_chunk, y_chunk = self._check_data(X_chunk, y_chunk)

        if not hasattr(self.model, "predict"):
            raise TypeError("The model or pipeline must support predict().")

        predictions = np.asarray(self.model.predict(X_chunk))

        if predictions.ndim != 1 or len(predictions) != len(y_chunk):
            raise ValueError("The model returned predictions with an invalid shape.")

        # Calculate accuracy for the current chunk.
        correct = int(np.sum(predictions == y_chunk))
        samples = len(y_chunk)
        accuracy = correct / samples

        # Update cumulative accuracy.
        self.total_correct += correct
        self.total_samples += samples

        cumulative_accuracy = (
            self.total_correct / self.total_samples
        )

        # Store the results in a dictionary.
        results = {
            "chunk": len(self.chunk_metrics) + 1,
            "samples": samples,
            "correct": correct,
            "accuracy": accuracy,
            "cumulative_accuracy": cumulative_accuracy
        }

        # Update an optional streaming metrics object.
        if self.metrics is not None:
            self.metrics.update(y_chunk, predictions)
            results["metrics"] = self.metrics.result()

        self.chunk_metrics.append(results)

        self.last_chunk_memory = (
            X_chunk.nbytes + y_chunk.nbytes + predictions.nbytes
        )

        return results

    def get_metrics(self):
        """Return the recorded results from all chunks."""
        return self.chunk_metrics.copy()

    def get_cumulative_accuracy(self):
        """Return accuracy across all scored chunks."""
        if self.total_samples == 0:
            return 0.0

        return self.total_correct / self.total_samples

    def memory_footprint(self):
        """Estimate memory used by the trainer and its model."""

        # Start with the memory used by the trainer's stored results.
        total_memory = sys.getsizeof(self.chunk_metrics)

        for result in self.chunk_metrics:
            total_memory += sys.getsizeof(result)

        # Include the latest input chunk.
        total_memory += self.last_chunk_memory

        # Include NumPy arrays stored directly in the model.
        if hasattr(self.model, "__dict__"):
            for value in vars(self.model).values():
                if isinstance(value, np.ndarray):
                    total_memory += value.nbytes
                elif isinstance(value, list):
                    total_memory += sys.getsizeof(value)

        return {
            "last_chunk_bytes": self.last_chunk_memory,
            "tracked_memory_bytes": total_memory
        }

    def reset_metrics(self):
        """Clear the performance results and cumulative counts."""
        self.chunk_metrics = []
        self.total_correct = 0
        self.total_samples = 0
        self.last_chunk_memory = 0

        if self.metrics is not None:
            self.metrics.reset()
