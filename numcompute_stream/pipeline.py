
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

        # Check that all intermediate steps can transform data.
        for name, step in steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(
                    f"Intermediate step '{name}' must have transform()."
                )

        self.steps = steps
        self._is_fitted = False

    @property
    def named_steps(self):
        """Return pipeline steps as a dictionary."""
        return dict(self.steps)

    def partial_fit(self, X_chunk, y_chunk):
        """Update transformers and the model using one data chunk."""
        X_chunk = np.asarray(X_chunk)
        y_chunk = np.asarray(y_chunk)

        if X_chunk.ndim != 2:
            raise ValueError("X_chunk must be a 2D array.")

        if y_chunk.ndim != 1:
            raise ValueError("y_chunk must be a 1D array.")

        if X_chunk.shape[0] == 0:
            raise ValueError("The data chunk cannot be empty.")

        if X_chunk.shape[0] != len(y_chunk):
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")

        X_current = X_chunk

        # Update and apply each preprocessing step.
        for name, step in self.steps[:-1]:
            if not hasattr(step, "partial_fit"):
                raise TypeError(
                    f"Step '{name}' must support partial_fit()."
                )

            step.partial_fit(X_current)
            X_current = step.transform(X_current)

        # Update the final model with the transformed data.
        model_name, model = self.steps[-1]

        if not hasattr(model, "partial_fit"):
            raise TypeError(
                f"Final step '{model_name}' must support partial_fit()."
            )

        model.partial_fit(X_current, y_chunk)

        self._is_fitted = True

        return self

    def fit(self, X, y):
        """Fit the complete pipeline using a batch of data."""
        X_current = np.asarray(X)
        y = np.asarray(y)

        if X_current.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if X_current.shape[0] == 0:
            raise ValueError("The training data cannot be empty.")

        if X_current.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows.")

        # Fit and transform the preprocessing steps.
        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(
                    f"Step '{name}' must support fit() and transform()."
                )

            if hasattr(step, "fit_transform"):
                X_current = step.fit_transform(X_current)
            else:
                step.fit(X_current)
                X_current = step.transform(X_current)

        # Fit the final model.
        model_name, model = self.steps[-1]

        if not hasattr(model, "fit"):
            raise TypeError(
                f"Final step '{model_name}' must support fit()."
            )

        model.fit(X_current, y)

        self._is_fitted = True

        return self

    def transform(self, X):
        """Apply all pipeline transformations to input data."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")

        X_current = np.asarray(X)

        if X_current.ndim != 2:
            raise ValueError("X must be a 2D array.")

        # Apply only the preprocessing steps.
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)

        # The final step must also be a transformer to return
        # transformed features from the complete pipeline.
        name, last_step = self.steps[-1]

        if not hasattr(last_step, "transform"):
            raise AttributeError(
                f"Final step '{name}' does not support transform()."
            )

        return last_step.transform(X_current)

    def predict(self, X):
        """Generate predictions using the fitted pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")

        X_current = np.asarray(X)

        if X_current.ndim != 2:
            raise ValueError("X must be a 2D array.")

        # Apply the preprocessing steps in order.
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)

        # Use the final model to make predictions.
        model_name, model = self.steps[-1]

        if not hasattr(model, "predict"):
            raise AttributeError(
                f"Final step '{model_name}' does not support predict()."
            )

        return model.predict(X_current)

    def fit_transform(self, X, y=None):
        """Fit and transform a pipeline containing only transformers."""
        if not hasattr(self.steps[-1][1], "transform"):
            raise AttributeError(
                "The final step must support transform() for fit_transform()."
            )

        self.fit(X, y)

        return self.transform(X)
