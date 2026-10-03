"""Shared split-conformal estimator lifecycle and finite-sample calibration."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone, is_regressor
from sklearn.model_selection import train_test_split
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, validate_data
from typing_extensions import Self

from ._typing import FeatureMatrix


def validate_alpha(alpha: float) -> float:
    """Validate a scalar miscoverage probability (never silently clip it).

    Parameters
    ----------
    alpha
        Miscoverage probability strictly between zero and one.
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite scalar in (0, 1)")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite scalar in (0, 1)")
    return float(alpha)


class BaseConformalPredictor(BaseEstimator, ABC):
    """Wrap a cloned estimator with held-out, unweighted split calibration.

    ``prediction_mode='conformal'`` preserves the original tuple-returning API.
    Use ``'point'`` for sklearn scorers, model selection and ensemble tools.
    ``calibration_size`` and ``random_state`` control automatic splitting.
    Put learned preprocessing inside the wrapped estimator's Pipeline so that
    calibration observations are excluded from all fitting steps.

    Fitted attributes
    -----------------
    estimator_ is the fitted clone; n_features_in_ and optional feature_names_in_
    describe its inputs. Classifiers additionally expose classes_. Calibration
    creates immutable calibration_scores_ and sorted_calibration_scores_, with
    n_calibration_ observations. calibration_non_conformity and n_calib retain
    the legacy aliases. Threshold queries record alpha_used_, quantile_level_
    (the finite-sample rank divided by sample count), and q_level_ (the threshold).
    These attributes are removed when their fitted or calibrated state expires.
    """

    # Attributes exist only after fitting, calibration, or a threshold query.
    estimator_: BaseEstimator
    classes_: NDArray[Any]
    n_features_in_: int
    feature_names_in_: NDArray[Any]
    calibration_scores_: NDArray[np.float64]
    sorted_calibration_scores_: NDArray[np.float64]
    calibration_non_conformity: NDArray[np.float64]
    n_calibration_: int
    n_calib: int
    alpha_used_: float
    q_level_: float
    quantile_level_: float

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        auto_calibrate: bool = False,
        tts_kwargs: Optional[dict[str, Any]] = None,
        calibration_size: Union[float, int] = 0.2,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        prediction_mode: Literal["conformal", "point"] = "conformal",
    ) -> None:
        """Configure the wrapped estimator and held-out calibration workflow.

        Parameters
        ----------
        estimator
            Cloneable sklearn estimator; classifiers also require predict_proba.
        auto_calibrate
            Reserve held-out calibration data during fit; defaults to False.
        tts_kwargs
            Optional train_test_split overrides, applied only with auto_calibrate.
        calibration_size
            Calibration fraction or sample count; defaults to 0.2.
        random_state
            Seed or RandomState for reproducible automatic splits.
        prediction_mode
            Return a conformal tuple by default, or point predictions in point mode.
        """
        self.estimator = estimator
        self.auto_calibrate = auto_calibrate
        self.tts_kwargs = tts_kwargs
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.prediction_mode = prediction_mode

    def __sklearn_tags__(self) -> Tags:
        """Advertise sparse input support to sklearn validation and tooling."""
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def set_params(self, **params: Any) -> Self:
        """Set sklearn parameters and invalidate fitted and calibrated state.

        Parameters
        ----------
        params
            Sklearn parameters, including nested estimator parameters.
        """
        result = super().set_params(**params)
        if params:
            self._clear_calibration()
            for name in (
                "estimator_",
                "classes_",
                "method_",
                "feature_names_in_",
                "n_features_in_",
            ):
                self.__dict__.pop(name, None)
        return result

    @property
    def is_calibrated_(self) -> bool:
        """Report whether calibration scores are available."""
        return hasattr(self, "calibration_scores_")

    @is_calibrated_.setter
    def is_calibrated_(self, value: bool) -> None:
        """Reset calibration or import scores from the legacy subclass protocol."""
        # Compatibility for subclasses using the original calibration protocol.
        if not value:
            self._clear_calibration()
        elif not hasattr(self, "calibration_scores_"):
            self.calibration_scores_ = np.asarray(self.calibration_non_conformity)

    def _clear_calibration(self) -> None:
        """Remove cached scores, sample counts, and threshold diagnostics."""
        for name in (
            "calibration_scores_",
            "sorted_calibration_scores_",
            "calibration_non_conformity",
            "n_calib",
            "n_calibration_",
            "alpha_used_",
            "q_level_",
            "quantile_level_",
        ):
            self.__dict__.pop(name, None)

    def fit(
        self,
        X: FeatureMatrix,
        y: ArrayLike,
        auto_calibrate: Optional[bool] = None,
        tts_kwargs: Optional[dict[str, Any]] = None,
        *,
        sample_weight: Optional[ArrayLike] = None,
        **fit_params: Any,
    ) -> Self:
        """Fit a fresh clone; optionally reserve an independent calibration split.

                Extra fit parameters are forwarded unchanged to the underlying estimator.
                ``sample_weight`` and step-qualified ``*__sample_weight`` vectors are
                validated and sliced with an automatic split. Other metadata is passed
                unchanged. Calibration remains unweighted.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y
            One target per sample; classification labels follow the fitted classes.
        auto_calibrate
            Reserve an independent calibration split when fitting.
        tts_kwargs
            Optional train_test_split overrides; requires automatic calibration.
        sample_weight
            Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
        fit_params
            Estimator-specific fit metadata forwarded to the cloned estimator.
        """
        # Invalidate before attempting a refit, including when fitting fails.
        self._clear_calibration()
        for name in ("estimator_", "classes_", "feature_names_in_", "n_features_in_"):
            self.__dict__.pop(name, None)
        if self.prediction_mode not in ("conformal", "point"):
            raise ValueError("prediction_mode must be 'conformal' or 'point'")
        if auto_calibrate is None:
            auto_calibrate = self.auto_calibrate
        if tts_kwargs is None:
            tts_kwargs = self.tts_kwargs if auto_calibrate else None
        if not isinstance(auto_calibrate, (bool, np.bool_)):
            raise ValueError("auto_calibrate must be boolean")
        if tts_kwargs is not None and not auto_calibrate:
            raise ValueError("tts_kwargs requires auto_calibrate=True")
        X_checked, y_checked = validate_data(
            self,
            X,
            y,
            accept_sparse=("csr", "csc"),
            dtype=None,
            y_numeric=is_regressor(self),
        )
        # Keep DataFrames for column-selecting pipelines, while validating above.
        X_fit = X if hasattr(X, "iloc") else X_checked
        # Pipeline fit keywords retain their names; only known per-row weights
        # are sliced. Arbitrary metadata may be scalar or estimator-specific.
        weight_params = {
            name: value
            for name, value in fit_params.items()
            if name.endswith("__sample_weight") and value is not None
        }
        if sample_weight is not None:
            weight_params["sample_weight"] = sample_weight
        for name, value in weight_params.items():
            weights = np.asarray(value, dtype=float)
            if (
                weights.shape != (len(y_checked),)
                or not np.isfinite(weights).all()
                or (weights < 0).any()
            ):
                raise ValueError(f"{name} must be finite, nonnegative, and match y")
            if not weights.any():
                raise ValueError(f"{name} must contain at least one non-zero weight")
            weight_params[name] = weights
        if auto_calibrate:
            options = {
                "test_size": self.calibration_size,
                "random_state": self.random_state,
            }
            options.update(tts_kwargs or {})
            # Split indices to retain DataFrame column names and slice weights.
            train, calib = train_test_split(np.arange(len(y_checked)), **options)

            def take(idx: NDArray[np.integer[Any]]) -> FeatureMatrix:
                """Select split rows while preserving DataFrame columns or sparse storage."""
                return X_fit.iloc[idx] if hasattr(X_fit, "iloc") else X_fit[idx]

            X_train, X_calib = take(train), take(calib)
            y_train, y_calib = y_checked[train], y_checked[calib]
            for name, weights in weight_params.items():
                if weights[train].sum() <= 0:
                    raise ValueError(f"{name} must have positive training weight")
                fit_params[name] = weights[train]
        else:
            X_train, y_train = X_fit, y_checked
            fit_params.update(weight_params)
        estimator = clone(self.estimator)
        self._validate_estimator(estimator)
        estimator.fit(X_train, y_train, **fit_params)
        self.estimator_ = estimator
        if hasattr(estimator, "classes_"):
            self.classes_ = np.asarray(estimator.classes_).copy()
        if auto_calibrate:
            self.calibrate(X_calib, y_calib)
        return self

    def _validate_estimator(self, estimator: BaseEstimator) -> None:
        """Require the prediction methods needed by this wrapper."""
        for method in ("fit", "predict"):
            if not callable(getattr(estimator, method, None)):
                raise TypeError(f"estimator must implement {method}")

    def _validate_X(self, X: FeatureMatrix) -> FeatureMatrix:
        """Check fitted state and feature layout, preserving DataFrame columns."""
        check_is_fitted(self, "estimator_")
        checked = validate_data(
            self, X, reset=False, accept_sparse=("csr", "csc"), dtype=None
        )
        return X if hasattr(X, "iloc") else checked

    def _validate_calibration(
        self, X: FeatureMatrix, y: ArrayLike, *, numeric: bool = False
    ) -> tuple[FeatureMatrix, NDArray[Any]]:
        """Validate aligned targets and features; optionally require finite numbers."""
        X = self._validate_X(X)
        from sklearn.utils.validation import check_consistent_length, column_or_1d

        y = column_or_1d(y)
        check_consistent_length(X, y)
        if numeric:
            y = np.asarray(y, dtype=float)
            if not np.isfinite(y).all():
                raise ValueError("calibration targets must be finite")
        return X, y

    def _store_scores(self, scores: ArrayLike) -> None:
        """Store immutable finite scores and their sorted order for repeated queries."""
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError("calibration scores must be a nonempty finite vector")
        if self.is_calibrated_:
            warnings.warn(
                "The estimator is already calibrated; replacing calibration scores.",
                UserWarning,
                stacklevel=2,
            )
        self._clear_calibration()
        self.calibration_scores_ = scores.copy()
        self.calibration_scores_.flags.writeable = False
        self.sorted_calibration_scores_ = np.sort(scores)
        self.sorted_calibration_scores_.flags.writeable = False
        self.calibration_non_conformity = self.calibration_scores_
        self.n_calibration_ = self.n_calib = len(scores)

    def _threshold(self, alpha: float) -> float:
        """Select the finite-sample order statistic, allowing an infinite threshold."""
        alpha = validate_alpha(alpha)
        if not self.is_calibrated_:
            raise RuntimeError(
                "The estimator must be calibrated. Call calibrate with held-out data."
            )
        rank = int(np.ceil((self.n_calibration_ + 1) * (1 - alpha)))
        self.alpha_used_ = alpha
        self.quantile_level_ = rank / self.n_calibration_
        if rank > self.n_calibration_:
            warnings.warn(
                "Quantile value requires an infinite threshold: calibration set is too small for alpha.",
                UserWarning,
                stacklevel=2,
            )
            self.q_level_ = np.inf
        else:
            self.q_level_ = float(self.sorted_calibration_scores_[rank - 1])
        return self.q_level_

    @abstractmethod
    def calibrate(self, X: FeatureMatrix, y: ArrayLike) -> Self:
        """Replace calibration scores using data independent of model fitting.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y
            One target per sample; classification labels follow the fitted classes.
        """
