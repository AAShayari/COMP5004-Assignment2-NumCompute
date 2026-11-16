
"""Pipeline for processing data in streaming chunks."""

import numpy as np


class Pipeline:
    """Chain preprocessing steps and a final machine learning model."""

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline steps cannot be empty.")

        names = [name for name, step in steps]

        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique.")

        for name, step in steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(
                    f"Preprocessing step '{name}' must support "
                    "fit() and transform()."
                )

        model_name, model = steps[-1]

        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError(
                f"Final step '{model_name}' must support fit() and predict()."
            )

        if not hasattr(model, "partial_fit"):
            raise TypeError(
                f"Final step '{model_name}' must support partial_fit()."
            )

        self.steps = steps

        # Store raw data until the next model retraining.
        self.buffer_X = []
        self.buffer_y = []

        self._is_fitted = False
        self.n_features_in_ = None

    @property
    def named_steps(self):
        """Return pipeline steps as a dictionary."""
        return dict(self.steps)

    @property
    def buffered_samples_(self):
        """Return the number of samples waiting for retraining."""
        if not self.buffer_X:
            return 0

        return sum(chunk.shape[0] for chunk in self.buffer_X)

    def _check_data(self, X, y=None):
        """Validate input features and optional target values."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] == 0:
            raise ValueError("The data cannot be empty.")

        if self.n_features_in_ is not None:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    "The number of input features must remain consistent."
                )

        if y is not None:
            y = np.asarray(y)

            if y.ndim != 1:
                raise ValueError("y must be a 1D array.")

            if X.shape[0] != len(y):
                raise ValueError(
                    "X and y must have the same number of rows."
                )

        return X, y

    def _transform_features(self, X):
        """Apply all preprocessing steps in their original order."""
        X_current = np.asarray(X)

        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)

        return X_current

    def _fit_transformers(self, X):
        """Fit preprocessing steps and transform the training data."""
        X_current = np.asarray(X)

        for name, step in self.steps[:-1]:
            step.fit(X_current)
            X_current = step.transform(X_current)

        return X_current

    def partial_fit(self, X_chunk, y_chunk):
        """
        Buffer raw data chunks and retrain when the buffer is full.

        The most recently trained model remains available for prediction
        while new samples are being collected.
        """
        X_chunk, y_chunk = self._check_data(X_chunk, y_chunk)

        # Remember the original number of input features.
        if self.n_features_in_ is None:
            self.n_features_in_ = X_chunk.shape[1]

        # Store copies of the original, untransformed data.
        self.buffer_X.append(X_chunk.copy())
        self.buffer_y.append(y_chunk.copy())

        model_name, model = self.steps[-1]

        # Use the final model's buffer size as the retraining threshold.
        buffer_size = getattr(model, "buffer_size", 1)

        if not isinstance(buffer_size, (int, np.integer)):
            raise ValueError("Model buffer_size must be a positive integer.")

        if buffer_size < 1:
            raise ValueError("Model buffer_size must be a positive integer.")

        # Wait until enough samples have accumulated.
        if self.buffered_samples_ < buffer_size:
            return self

        # Combine all pending raw data.
        X_all = np.concatenate(self.buffer_X, axis=0)
        y_all = np.concatenate(self.buffer_y, axis=0)

        # Refit transformers on the complete pending buffer.
        X_transformed = self._fit_transformers(X_all)

        # Retrain the model using the consistently transformed data.
        # fit() resets the model's own internal buffer.
        model.fit(X_transformed, y_all)

        # Clear the pipeline buffer after successful retraining.
        self.buffer_X = []
        self.buffer_y = []

        self._is_fitted = True

        return self

    def fit(self, X, y):
        """Fit the complete pipeline using a batch of data."""
        X, y = self._check_data(X, y)

        self.n_features_in_ = X.shape[1]

        # Fit and apply preprocessing steps.
        X_transformed = self._fit_transformers(X)

        # Fit the final model.
        model_name, model = self.steps[-1]
        model.fit(X_transformed, y)

        # Reset any pending streaming data.
        self.buffer_X = []
        self.buffer_y = []

        self._is_fitted = True

        return self

    def transform(self, X):
        """Apply the fitted preprocessing steps to input data."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")

        X, _ = self._check_data(X)

        return self._transform_features(X)

    def predict(self, X):
        """Generate predictions using the most recently trained model."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")

        X, _ = self._check_data(X)

        X_transformed = self._transform_features(X)

        model_name, model = self.steps[-1]

        return model.predict(X_transformed)

    def fit_transform(self, X, y):
        """Fit the pipeline and return transformed features."""
        self.fit(X, y)

        return self.transform(X)

