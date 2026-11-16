
"""Tools for training and evaluating models with streaming data."""

import sys
import numpy as np


class StreamTrainer:
    """Train and evaluate a model or pipeline using data chunks."""

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
        self.metrics = metrics

        # Store the results of each chunk.
        self.chunk_metrics = []

        # Track cumulative accuracy.
        self.total_correct = 0
        self.total_samples = 0

        # Track the number of processed chunks.
        self.training_chunks = 0
        self.evaluation_chunks = 0

        # Track the memory used by the latest chunk.
        self.last_chunk_memory = 0

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------

    def _check_data(self, X, y):
        """Check the dimensions and sizes of the input data."""

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

    # --------------------------------------------------
    # Memory estimation
    # --------------------------------------------------

    def memory_footprint(self):
        """
        Estimate memory used by the latest chunk and model buffers.

        This is an approximate measurement, not total Python
        process memory.
        """

        model_memory = sys.getsizeof(self.model)

        # Count NumPy arrays stored directly by the model.
        if hasattr(self.model, "__dict__"):

            for value in vars(self.model).values():

                if isinstance(value, np.ndarray):
                    model_memory += value.nbytes

                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, np.ndarray):
                            model_memory += item.nbytes

        tracked_memory = (
            self.last_chunk_memory + model_memory
        )

        return {
            "last_chunk_bytes": self.last_chunk_memory,
            "model_memory_bytes": model_memory,
            "tracked_memory_bytes": tracked_memory
        }

    def _record_chunk(self, results):
        """Save the results and memory estimate for a chunk."""

        memory = self.memory_footprint()

        results["memory_bytes"] = memory["last_chunk_bytes"]
        results["model_memory_bytes"] = memory["model_memory_bytes"]
        results["tracked_memory_bytes"] = (
            memory["tracked_memory_bytes"]
        )

        self.chunk_metrics.append(results)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def fit_chunk(self, X_chunk, y_chunk):
        """Train the model using one incoming data chunk."""

        X_chunk, y_chunk = self._check_data(
            X_chunk, y_chunk
        )

        if not hasattr(self.model, "partial_fit"):
            raise TypeError(
                "The model or pipeline must support partial_fit()."
            )

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

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    def score_chunk(self, X_chunk, y_chunk):
        """Evaluate the model using one incoming data chunk."""

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

        if predictions.ndim != 1:
            raise ValueError(
                "Predictions must be a 1D array."
            )

        if len(predictions) != len(y_chunk):
            raise ValueError(
                "The model returned an incorrect number of predictions."
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

        # Update optional streaming classification metrics.
        if self.metrics is not None:

            self.metrics.update(y_chunk, predictions)

            results["metrics"] = self.metrics.result()

        self._record_chunk(results)

        return results

    # --------------------------------------------------
    # Results and reset
    # --------------------------------------------------

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

        self.training_chunks = 0
        self.evaluation_chunks = 0

        self.last_chunk_memory = 0

        if self.metrics is not None:
            self.metrics.reset()
