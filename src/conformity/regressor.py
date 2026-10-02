"""Split conformal regression with exact finite-sample order statistics."""

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from .base import BaseConformalPredictor


class ConformalRegressor(RegressorMixin, BaseConformalPredictor):
    """Symmetric absolute-residual prediction intervals for single-output regression.

    See BaseConformalPredictor for constructor and splitting options. Infinite
    intervals are required when alpha < 1 / (n_calibration + 1). Coverage is
    marginal under exchangeability, not conditional on each feature value.
    """

    def _point(self, X):
        pred = np.asarray(self.estimator_.predict(X), dtype=float)
        if pred.shape != (X.shape[0],) or not np.isfinite(pred).all():
            raise ValueError(
                "estimator.predict must return one finite value per sample"
            )
        return pred

    def calibrate(self, X, y):
        X, y = self._validate_calibration(X, y, numeric=True)
        self._store_scores(np.abs(y - self._point(X)))
        return self

    def predict_point(self, X):
        """Return point predictions without requiring calibration."""
        return self._point(self._validate_X(X))

    def predict_interval(self, X, alpha=0.05):
        """Return [lower, upper] bounds with shape (n_samples, 2)."""
        X = self._validate_X(X)
        threshold = self._threshold(alpha)
        point = self._point(X)
        return np.column_stack((point - threshold, point + threshold))

    def predict(self, X, alpha=0.05):
        """Return (point, intervals), or points in prediction_mode='point'."""
        if self.prediction_mode == "point":
            return self.predict_point(X)
        X = self._validate_X(X)
        threshold = self._threshold(alpha)
        point = self._point(X)
        return point, np.column_stack((point - threshold, point + threshold))

    def score(self, X, y, sample_weight=None):
        """Return point-prediction R², regardless of prediction_mode."""
        return r2_score(y, self.predict_point(X), sample_weight=sample_weight)

    def evaluate(self, X, y, alpha=0.05, *, sample_weight=None):
        """Return point accuracy, coverage, mean width, and interval score."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from .metrics import (
            prediction_interval_coverage,
            prediction_interval_width,
            interval_score,
        )

        X = self._validate_X(X)
        threshold = self._threshold(alpha)
        point = self._point(X)
        intervals = np.column_stack((point - threshold, point + threshold))
        return {
            "r2": r2_score(y, point, sample_weight=sample_weight),
            "mae": mean_absolute_error(y, point, sample_weight=sample_weight),
            "mse": mean_squared_error(y, point, sample_weight=sample_weight),
            "coverage": prediction_interval_coverage(
                y, intervals, sample_weight=sample_weight
            ),
            "mean_width": prediction_interval_width(
                intervals, sample_weight=sample_weight
            ),
            "interval_score": interval_score(
                y, intervals, alpha, sample_weight=sample_weight
            ),
        }
