
"""
Tools for training and evaluating models with streaming data.
"""

import sys
import numpy as np


class StreamTrainer:
    """
    Train and evaluate a model or pipeline using incoming data chunks.

    Records training history, evaluation metrics, cumulative accuracy
    and estimated memory usage.
    """

    def __init__(self, model=None, pipeline=None, metrics=None):
        if model is not None and pipeline is not None:
            raise ValueError(
                "Provide either a model or a pipeline, not both."
            )

        if model is None and pipeline is None:
            raise ValueError(
                "A model or pipeline must be provided."
            )

        self.model = model if model is not None else pipeline
        self.pipeline = pipeline
        self.metrics = metrics

        # Store training and evaluation results.
        self.chunk_metrics = []

        # Keep track of cumulative accuracy.
        self.total_correct = 0
        self.total_samples = 0

        # Keep track of the latest chunk's memory usage.
        self.last_chunk_memory = 0

        # Count training and evaluation chunks separately.
        self.training_chunks = 0
        self.evaluation_chunks = 0

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
            raise ValueError(
                "X and y must have the same number of rows."
            )

        return X, y

    def _estimate_size(self, obj, seen=None):
        """
        Estimate the memory occupied by an object and its contents.

        This is an approximate measurement, not the total process
        memory used by Python.
        """
        if seen is None:
            seen = set()

        object_id = id(obj)

        if object_id in seen:
            return 0

        seen.add(object_id)

        if isinstance(obj, np.ndarray):
            return obj.nbytes

        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            for key, value in obj.items():
                size += self._estimate_size(key, seen)
                size += self._estimate_size(value, seen)

        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                size += self._estimate_size(item, seen)

        elif hasattr(obj, "__dict__"):
            size += self._estimate_size(vars(obj), seen)

        return size

    def memory_footprint(self):
        """
        Return an estimate of the memory tracked by the trainer.

        Returns
        -------
        dict
            Latest chunk memory and estimated tracked memory in bytes.
        """
        history_memory = self._estimate_size(self.chunk_metrics)
        model_memory = self._estimate_size(self.model)

        tracked_memory = (
            history_memory
            + model_memory
            + self.last_chunk_memory
        )

        return {
            "last_chunk_bytes": self.last_chunk_memory,
            "tracked_memory_bytes": tracked_memory
        }

    def _record_chunk(self, results):
        """Add a chunk's results and memory estimate to the history."""
        self.chunk_metrics.append(results)

        memory = self.memory_footprint()

        results["memory_bytes"] = memory["last_chunk_bytes"]
        results["tracked_memory_bytes"] = (
            memory["tracked_memory_bytes"]
        )

    def fit_chunk(self, X_chunk, y_chunk):
        """
        Train the model or pipeline using one incoming data chunk.

        The model must support partial_fit().
        """
        X_chunk, y_chunk = self._check_data(
            X_chunk, y_chunk
        )

        if not hasattr(self.model, "partial_fit"):
            raise TypeError(
                "The model or pipeline must support partial_fit()."
            )

        # Train the model using the incoming chunk.
        self.model.partial_fit(X_chunk, y_chunk)

        self.training_chunks += 1

        self.last_chunk_memory = (
            X_chunk.nbytes + y_chunk.nbytes
        )

        results = {
            "chunk": len(self.chunk_metrics) + 1,
            "type": "training",
            "samples": len(y_chunk),
            "training_chunks": self.training_chunks,
            "evaluation_chunks": self.evaluation_chunks
        }

        self._record_chunk(results)

        return self

    def score_chunk(self, X_chunk, y_chunk):
        """
        Evaluate the model using one incoming data chunk.

        Records chunk accuracy, cumulative accuracy, optional
        classification metrics and estimated memory usage.
        """
        X_chunk, y_chunk = self._check_data(
            X_chunk, y_chunk
        )

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "The model or pipeline must support predict()."
            )

        predictions = np.asarray(
            self.model.predict(X_chunk)
        )

        if predictions.ndim != 1 or len(predictions) != len(y_chunk):
            raise ValueError(
                "The model returned predictions with an invalid shape."
            )

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

        self.evaluation_chunks += 1

        self.last_chunk_memory = (
            X_chunk.nbytes
            + y_chunk.nbytes
            + predictions.nbytes
        )

        results = {
            "chunk": len(self.chunk_metrics) + 1,
            "type": "evaluation",
            "samples": samples,
            "correct": correct,
            "accuracy": accuracy,
            "cumulative_accuracy": cumulative_accuracy,
            "training_chunks": self.training_chunks,
            "evaluation_chunks": self.evaluation_chunks
        }

        # Update an optional streaming metrics object.
        if self.metrics is not None:
            self.metrics.update(y_chunk, predictions)
            results["metrics"] = self.metrics.result()

        self._record_chunk(results)

        return results

    def get_metrics(self):
        """Return the recorded results from all chunks."""
        return self.chunk_metrics.copy()

    def get_cumulative_accuracy(self):
        """Return accuracy across all evaluated chunks."""
        if self.total_samples == 0:
            return 0.0

        return self.total_correct / self.total_samples

    def reset_metrics(self):
        """Clear the recorded results and cumulative counts."""
        self.chunk_metrics = []

        self.total_correct = 0
        self.total_samples = 0

        self.last_chunk_memory = 0

        self.training_chunks = 0
        self.evaluation_chunks = 0

        if self.metrics is not None:
            self.metrics.reset()
