"""Split conformal regression with exact finite-sample order statistics."""

from __future__ import annotations

import warnings
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from typing_extensions import Self

from ._typing import FeatureMatrix, FloatArray, TrainTestSplitKwargs
from .base import BaseConformalPredictor


class ConformalRegressor(RegressorMixin, BaseConformalPredictor):
    """Split-conformal prediction intervals for single-output regression.

    ``interval_method="symmetric"`` uses one absolute-residual quantile for
    equal-width corrections. ``interval_method="asymmetric"`` calibrates
    lower and upper signed residuals separately, allocating alpha / 2 to each
    tail. By the union bound, the latter has marginal coverage at least
    1 - alpha under exchangeability, with no symmetry assumption on residuals.

    See BaseConformalPredictor for splitting options. Symmetric intervals
    require alpha >= 1 / (n_calibration + 1) for a finite threshold; the
    asymmetric method requires alpha >= 2 / (n_calibration + 1). Coverage is
    marginal under exchangeability, not conditional on each feature value.

    Attributes
    ----------
    lower_q_level_ : float
        Most recent lower correction for asymmetric intervals.
    upper_q_level_ : float
        Most recent upper correction for asymmetric intervals.
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
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
    ...     Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
    ...     auto_calibrate=False,
    ... )
    >>> _ = model.fit(X_train, y_train)
    >>> _ = model.calibrate(X_calib, y_calib)
    >>> points, intervals = model.predict(X_test, alpha=0.1, return_interval=True)
    >>> points.shape, intervals.shape
    ((30,), (30, 2))
    >>> model.n_calibration_
    30
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        auto_calibrate: bool = True,
        tts_kwargs: Optional[TrainTestSplitKwargs] = None,
        interval_method: Literal["symmetric", "asymmetric"] = "symmetric",
    ) -> None:
        """Configure the regressor, split and interval calibration method.

        Parameters
        ----------
        estimator : BaseEstimator
            Cloneable sklearn regressor or preprocessing-and-regression pipeline.
        auto_calibrate : bool
            Reserve held-out calibration data during fit; defaults to True.
        tts_kwargs : TrainTestSplitKwargs or None
            train_test_split options for automatic calibration. By default,
            test_size=0.2, shuffle=True and stratify=None.
        interval_method : {"symmetric", "asymmetric"}
            Symmetric absolute-residual calibration (default), or separate signed
            residual quantiles with alpha / 2 allocated to each tail.

        """
        super().__init__(
            estimator,
            auto_calibrate=auto_calibrate,
            tts_kwargs=tts_kwargs,
        )
        self.interval_method = interval_method

    def _validate_estimator(self, estimator: BaseEstimator) -> None:
        """Validate the shared estimator contract and interval configuration."""
        super()._validate_estimator(estimator)
        if not isinstance(self.interval_method, str) or self.interval_method not in (
            "symmetric",
            "asymmetric",
        ):
            raise ValueError("interval_method must be 'symmetric' or 'asymmetric'")

    def _clear_calibration(self) -> None:
        """Remove symmetric and one-sided calibration scores."""
        super()._clear_calibration()
        for name in (
            "sorted_lower_calibration_scores_",
            "sorted_upper_calibration_scores_",
            "lower_q_level_",
            "upper_q_level_",
        ):
            self.__dict__.pop(name, None)

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
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y : array-like of shape (n_samples,)
            One target per sample; classification labels follow the fitted classes.

        Returns
        -------
        self : object
            Predictor with replacement calibration scores.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.

        """
        X, y = self._validate_calibration(X, y, numeric=True)
        point = self._point(X)
        self._store_scores(np.abs(y - point))
        if self.interval_method == "asymmetric":
            lower_scores = np.sort(point - y)
            upper_scores = np.sort(y - point)
            lower_scores.flags.writeable = False
            upper_scores.flags.writeable = False
            self.sorted_lower_calibration_scores_ = lower_scores
            self.sorted_upper_calibration_scores_ = upper_scores
        return self

    def _interval_thresholds(self, alpha: float) -> tuple[float, float]:
        """Return lower and upper corrections with Bonferroni tail allocation."""
        if self.interval_method == "symmetric":
            threshold = self._threshold(alpha)
            return threshold, threshold
        from .base import validate_alpha

        alpha = validate_alpha(alpha)
        if not self.is_calibrated_:
            raise RuntimeError(
                "The estimator must be calibrated. Call calibrate with held-out data."
            )
        if not hasattr(self, "sorted_lower_calibration_scores_"):
            raise RuntimeError("Calibration scores do not match interval_method")
        rank = int(np.ceil((self.n_calibration_ + 1) * (1 - alpha / 2)))
        self.alpha_used_ = alpha
        self.quantile_level_ = rank / self.n_calibration_
        if rank > self.n_calibration_:
            warnings.warn(
                "One-sided quantiles require infinite interval bounds: calibration "
                "set is too small for alpha / 2.",
                UserWarning,
                stacklevel=2,
            )
            lower = upper = np.inf
        else:
            # A nonnegative correction keeps bounds ordered and only enlarges
            # each one-sided set of covered observations.
            lower = max(0.0, float(self.sorted_lower_calibration_scores_[rank - 1]))
            upper = max(0.0, float(self.sorted_upper_calibration_scores_[rank - 1]))
        self.lower_q_level_ = lower
        self.upper_q_level_ = upper
        self.q_level_ = max(lower, upper)
        return lower, upper

    def predict_point(self, X: FeatureMatrix) -> FloatArray:
        """Return point predictions without requiring calibration.

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Point predictions from the fitted estimator.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.

        """
        return self._point(self._validate_X(X))

    def predict_interval(self, X: FeatureMatrix, alpha: float = 0.05) -> FloatArray:
        """Return [lower, upper] bounds with shape (n_samples, 2).

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        alpha : float
            Miscoverage probability strictly between zero and one.

        Returns
        -------
        intervals : ndarray of shape (n_samples, 2)
            Closed lower and upper bounds; may be outward infinite.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.
        RuntimeError
            If conformal output is requested before calibration.

        """
        X = self._validate_X(X)
        lower, upper = self._interval_thresholds(alpha)
        point = self._point(X)
        return np.column_stack((point - lower, point + upper))

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
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        alpha : float
            Miscoverage probability strictly between zero and one.
        return_interval : bool
            Request conformal output alongside ordinary predictions.

        Returns
        -------
        predictions : ndarray or tuple of ndarray
            Point predictions, or (points, intervals) when return_interval=True.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.
        RuntimeError
            If conformal output is requested before calibration.

        """
        if not isinstance(return_interval, (bool, np.bool_)):
            raise ValueError("return_interval must be boolean")
        if not return_interval:
            return self.predict_point(X)
        X = self._validate_X(X)
        lower, upper = self._interval_thresholds(alpha)
        point = self._point(X)
        return point, np.column_stack((point - lower, point + upper))

    def score(
        self, X: FeatureMatrix, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> float:
        """Return point-prediction R², using point predictions.

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y : array-like of shape (n_samples,)
            One target per sample; classification labels follow the fitted classes.
        sample_weight : array-like of shape (n_samples,) or None
            Optional finite, nonnegative weights; fitting or evaluation only, never calibration.

        Returns
        -------
        score : float
            Point-prediction R² for regression or accuracy for classification.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.

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
        """Return point error, coverage, tail misses, width and interval score.

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y : array-like of shape (n_samples,)
            One target per sample; classification labels follow the fitted classes.
        alpha : float
            Miscoverage probability strictly between zero and one.
        sample_weight : array-like of shape (n_samples,) or None
            Optional finite, nonnegative weights; fitting or evaluation only, never calibration.

        Returns
        -------
        report : dict of str to float
            Keys are r2, mae, mse, coverage, lower_miss_rate,
            upper_miss_rate, mean_width, normalized_mean_width and
            interval_score. Normalized width divides mean width by the
            unweighted test-target IQR and is infinite when that IQR is zero.
            Tail miss rates are empirical diagnostics; for asymmetric
            intervals compare each with alpha / 2, allowing for finite
            test-sample variation.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.
        RuntimeError
            If conformal output is requested before calibration.

        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        from .metrics import (
            interval_score,
            prediction_interval_coverage,
            prediction_interval_lower_miss_rate,
            prediction_interval_upper_miss_rate,
            prediction_interval_width,
            normalized_prediction_interval_width,
        )

        X = self._validate_X(X)
        lower, upper = self._interval_thresholds(alpha)
        point = self._point(X)
        intervals = np.column_stack((point - lower, point + upper))
        return {
            "r2": r2_score(y, point, sample_weight=sample_weight),
            "mae": mean_absolute_error(y, point, sample_weight=sample_weight),
            "mse": mean_squared_error(y, point, sample_weight=sample_weight),
            "coverage": prediction_interval_coverage(
                y, intervals, sample_weight=sample_weight
            ),
            "lower_miss_rate": prediction_interval_lower_miss_rate(
                y, intervals, sample_weight=sample_weight
            ),
            "upper_miss_rate": prediction_interval_upper_miss_rate(
                y, intervals, sample_weight=sample_weight
            ),
            "mean_width": prediction_interval_width(
                intervals, sample_weight=sample_weight
            ),
            "normalized_mean_width": normalized_prediction_interval_width(
                y, intervals, sample_weight=sample_weight
            ),
            "interval_score": interval_score(
                y, intervals, alpha, sample_weight=sample_weight
            ),
        }
