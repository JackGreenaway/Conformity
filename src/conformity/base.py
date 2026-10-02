"""Shared split-conformal estimator lifecycle and finite-sample calibration."""

from abc import ABC, abstractmethod
from numbers import Real
import warnings

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data


def validate_alpha(alpha):
    """Validate a scalar miscoverage probability (never silently clip it)."""
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
    """

    def __init__(
        self,
        estimator,
        *,
        auto_calibrate=False,
        tts_kwargs=None,
        calibration_size=0.2,
        random_state=None,
        prediction_mode="conformal",
    ):
        self.estimator = estimator
        self.auto_calibrate = auto_calibrate
        self.tts_kwargs = tts_kwargs
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.prediction_mode = prediction_mode

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def set_params(self, **params):
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
    def is_calibrated_(self):
        return hasattr(self, "calibration_scores_")

    @is_calibrated_.setter
    def is_calibrated_(self, value):
        # Compatibility for subclasses using the original calibration protocol.
        if not value:
            self._clear_calibration()
        elif not hasattr(self, "calibration_scores_"):
            self.calibration_scores_ = np.asarray(self.calibration_non_conformity)

    def _clear_calibration(self):
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
        X,
        y,
        auto_calibrate=None,
        tts_kwargs=None,
        *,
        sample_weight=None,
        **fit_params,
    ):
        """Fit a fresh clone; optionally reserve an independent calibration split.

        Extra fit parameters are forwarded unchanged to the underlying estimator.
        ``sample_weight`` and step-qualified ``*__sample_weight`` vectors are
        validated and sliced with an automatic split. Other metadata is passed
        unchanged. Calibration remains unweighted.
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
            y_numeric=getattr(self, "_estimator_type", None) == "regressor",
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
                or weights.sum() <= 0
            ):
                raise ValueError(f"{name} must be finite, nonnegative, and match y")
            weight_params[name] = weights
        if auto_calibrate:
            options = {
                "test_size": self.calibration_size,
                "random_state": self.random_state,
            }
            options.update(tts_kwargs or {})
            # Split indices to retain DataFrame column names and slice weights.
            train, calib = train_test_split(np.arange(len(y_checked)), **options)

            def take(idx):
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

    def _validate_estimator(self, estimator):
        for method in ("fit", "predict"):
            if not callable(getattr(estimator, method, None)):
                raise TypeError(f"estimator must implement {method}")

    def _validate_X(self, X):
        check_is_fitted(self, "estimator_")
        checked = validate_data(
            self, X, reset=False, accept_sparse=("csr", "csc"), dtype=None
        )
        return X if hasattr(X, "iloc") else checked

    def _validate_calibration(self, X, y, *, numeric=False):
        X = self._validate_X(X)
        from sklearn.utils.validation import column_or_1d, check_consistent_length

        y = column_or_1d(y)
        check_consistent_length(X, y)
        if numeric:
            y = np.asarray(y, dtype=float)
            if not np.isfinite(y).all():
                raise ValueError("calibration targets must be finite")
        return X, y

    def _store_scores(self, scores):
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

    def _threshold(self, alpha):
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
    def calibrate(self, X, y):
        """Replace calibration scores using data independent of model fitting."""
