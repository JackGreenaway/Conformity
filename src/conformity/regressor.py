"""Split conformal regression with exact finite-sample order statistics."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from typing_extensions import Self

from ._typing import FeatureMatrix, FloatArray
from .base import BaseConformalPredictor


class ConformalRegressor(RegressorMixin, BaseConformalPredictor):
    """Symmetric absolute-residual prediction intervals for single-output regression.

    See BaseConformalPredictor for constructor and splitting options. Infinite
    intervals are required when alpha < 1 / (n_calibration + 1). Coverage is
    marginal under exchangeability, not conditional on each feature value.
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from conformity import ConformalRegressor
    >>> X, y = make_regression(
    ...     n_samples=120, n_features=3, noise=4, random_state=0
    ... )
    >>> X_dev, X_test, y_dev, y_test = train_test_split(
    ...     X, y, test_size=0.25, random_state=1
    ... )
    >>> X_train, X_calib, y_train, y_calib = train_test_split(
    ...     X_dev, y_dev, test_size=1 / 3, random_state=2
    ... )
    >>> model = ConformalRegressor(
    ...     make_pipeline(StandardScaler(), Ridge()), auto_calibrate=False
    ... )
    >>> _ = model.fit(X_train, y_train)
    >>> _ = model.calibrate(X_calib, y_calib)
    >>> points, intervals = model.predict(X_test, alpha=0.1, return_interval=True)
    >>> points.shape, intervals.shape
    ((30,), (30, 2))
    >>> model.n_calibration_
    30
    """

    def _point(self, X: FeatureMatrix) -> FloatArray:
        """Validate one finite numeric prediction per input sample."""
        pred = np.asarray(self.estimator_.predict(X), dtype=float)
        if pred.shape != (X.shape[0],) or not np.isfinite(pred).all():
            raise ValueError(
                "estimator.predict must return one finite value per sample"
            )
        return pred

    def calibrate(self, X: FeatureMatrix, y: ArrayLike) -> Self:
        """Replace held-out scores and return this predictor for method chaining.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y
            One target per sample; classification labels follow the fitted classes.
        """
        X, y = self._validate_calibration(X, y, numeric=True)
        self._store_scores(np.abs(y - self._point(X)))
        return self

    def predict_point(self, X: FeatureMatrix) -> FloatArray:
        """Return point predictions without requiring calibration.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        """
        return self._point(self._validate_X(X))

    def predict_interval(self, X: FeatureMatrix, alpha: float = 0.05) -> FloatArray:
        """Return [lower, upper] bounds with shape (n_samples, 2).

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        alpha
            Miscoverage probability strictly between zero and one.
        """
        X = self._validate_X(X)
        threshold = self._threshold(alpha)
        point = self._point(X)
        return np.column_stack((point - threshold, point + threshold))

    def predict(
        self,
        X: FeatureMatrix,
        alpha: float = 0.05,
        *,
        return_interval: bool = False,
    ) -> Union[FloatArray, tuple[FloatArray, FloatArray]]:
        """Return points or (points, intervals).

                ``return_interval=True`` requests intervals and can be
                passed through a sklearn Pipeline with metadata routing disabled.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        alpha
            Miscoverage probability strictly between zero and one.
        return_interval
            Request conformal output alongside ordinary predictions.
        """
        if not isinstance(return_interval, (bool, np.bool_)):
            raise ValueError("return_interval must be boolean")
        if not return_interval:
            return self.predict_point(X)
        X = self._validate_X(X)
        threshold = self._threshold(alpha)
        point = self._point(X)
        return point, np.column_stack((point - threshold, point + threshold))

    def score(
        self, X: FeatureMatrix, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> float:
        """Return point-prediction R², using point predictions.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y
            One target per sample; classification labels follow the fitted classes.
        sample_weight
            Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
        """
        return r2_score(y, self.predict_point(X), sample_weight=sample_weight)

    def evaluate(
        self,
        X: FeatureMatrix,
        y: ArrayLike,
        alpha: float = 0.05,
        *,
        sample_weight: Optional[ArrayLike] = None,
    ) -> dict[str, float]:
        """Return point accuracy, coverage, mean width, and interval score.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y
            One target per sample; classification labels follow the fitted classes.
        alpha
            Miscoverage probability strictly between zero and one.
        sample_weight
            Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        from .metrics import (
            interval_score,
            prediction_interval_coverage,
            prediction_interval_width,
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
