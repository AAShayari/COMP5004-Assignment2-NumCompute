
"""Pipeline for processing data in streaming chunks."""

import numpy as np


class Pipeline:
    """Apply preprocessing steps before training a machine learning model."""

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline steps cannot be empty.")

        names = [name for name, step in steps]

        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique.")

        for name, step in steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"Step '{name}' must have transform()."
                )

        self.steps = steps
        self._is_fitted = False

        # Store raw data until the next model retraining.
        self.buffer_X = []
        self.buffer_y = []

    @property
    def named_steps(self):
        """Return pipeline steps as a dictionary."""
        return dict(self.steps)

    @property
    def buffered_samples_(self):
        """Return the number of samples waiting in the buffer."""
        return sum(len(chunk) for chunk in self.buffer_X)

    def _check_data(self, X, y):
        """Check the input feature and target arrays."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X.shape[0] == 0:
            raise ValueError("The data cannot be empty.")

        if X.shape[0] != len(y):
            raise ValueError(
                "X and y must have the same number of rows."
            )

        return X, y

    def _transform_data(self, X):
        """Apply all fitted preprocessing steps."""
        X_current = X

        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)

        return X_current

    def partial_fit(self, X_chunk, y_chunk):
        """Collect chunks and retrain when the buffer is full."""
        X_chunk, y_chunk = self._check_data(
            X_chunk, y_chunk
        )

        if self.buffer_X:
            if X_chunk.shape[1] != self.buffer_X[0].shape[1]:
                raise ValueError(
                    "All chunks must have the same number of features."
                )

        self.buffer_X.append(X_chunk.copy())
        self.buffer_y.append(y_chunk.copy())

        model_name, model = self.steps[-1]

        # Use the model's configured buffer size.
        buffer_size = getattr(model, "buffer_size", 1)

        if self.buffered_samples_ >= buffer_size:

            # Combine all waiting raw data.
            X_all = np.concatenate(self.buffer_X, axis=0)
            y_all = np.concatenate(self.buffer_y, axis=0)

            X_current = X_all

            # Update and apply each preprocessing step.
            for name, step in self.steps[:-1]:

                if not hasattr(step, "partial_fit"):
                    raise TypeError(
                        f"Step '{name}' must support partial_fit()."
                    )

                step.partial_fit(X_current)
                X_current = step.transform(X_current)

            # Retrain the model using consistently transformed data.
            if not hasattr(model, "fit"):
                raise TypeError(
                    f"Final step '{model_name}' must support fit()."
                )

            model.fit(X_current, y_all)

            # Clear the buffer after successful retraining.
            self.buffer_X = []
            self.buffer_y = []

            self._is_fitted = True

        return self

    def fit(self, X, y):
        """Fit the complete pipeline using a batch of data."""
        X, y = self._check_data(X, y)

        X_current = X

        # Fit and transform the preprocessing steps.
        for name, step in self.steps[:-1]:

            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(
                    f"Step '{name}' must support fit() and transform()."
                )

            step.fit(X_current)
            X_current = step.transform(X_current)

        # Fit the final model.
        model_name, model = self.steps[-1]

        if not hasattr(model, "fit"):
            raise TypeError(
                f"Final step '{model_name}' must support fit()."
            )

        model.fit(X_current, y)

        self.buffer_X = []
        self.buffer_y = []
        self._is_fitted = True

        return self

    def transform(self, X):
        """Apply the fitted preprocessing steps."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        return self._transform_data(X)

    def predict(self, X):
        """Generate predictions using the fitted pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        X_current = self._transform_data(X)

        model_name, model = self.steps[-1]

        if not hasattr(model, "predict"):
            raise AttributeError(
                f"Final step '{model_name}' must support predict()."
            )

        return model.predict(X_current)
