"""Label-safe least ambiguous and adaptive prediction sets."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from typing_extensions import Self

from ._typing import FeatureMatrix, FloatArray
from .base import BaseConformalPredictor


class ConformalClassifier(ClassifierMixin, BaseConformalPredictor):
    """Split conformal classification using LAC or deterministic APS scores.

    ``method='lac'`` scores 1 - probability; ``'aps'`` scores cumulative
    descending probability through the candidate class. Ties use stable class
    order. ``predict_set`` returns a boolean matrix in ``classes_`` order, which
    works with any sklearn class labels. Empty prediction sets are permitted.
    """

    # Score method recorded when calibration succeeds.
    method_: Literal["lac", "aps"]

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        method: Literal["lac", "aps"] = "lac",
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
            Cloneable sklearn classifier implementing predict_proba.
        method
            Use lac (1 - probability) or aps (cumulative ranked probability).
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
        super().__init__(
            estimator,
            auto_calibrate=auto_calibrate,
            tts_kwargs=tts_kwargs,
            calibration_size=calibration_size,
            random_state=random_state,
            prediction_mode=prediction_mode,
        )
        self.method = method

    def _validate_estimator(self, estimator: BaseEstimator) -> None:
        """Require the prediction methods needed by this wrapper."""
        super()._validate_estimator(estimator)
        if self.method not in ("lac", "aps"):
            raise ValueError("method must be 'lac' or 'aps'")
        if not callable(getattr(estimator, "predict_proba", None)):
            raise TypeError("classification estimator must implement predict_proba")

    def _probabilities(self, X: FeatureMatrix) -> FloatArray:
        """Validate a probability matrix with one column per fitted class."""
        proba = np.asarray(self.estimator_.predict_proba(X), dtype=float)
        if proba.shape != (X.shape[0], len(self.classes_)):
            raise ValueError("predict_proba shape must match samples and classes_")
        if (
            not np.isfinite(proba).all()
            or (proba < 0).any()
            or (proba > 1).any()
            or not np.allclose(proba.sum(axis=1), 1, atol=1e-7)
        ):
            raise ValueError(
                "predict_proba must contain finite probabilities summing to one"
            )
        return proba

    def _candidate_scores(self, proba: FloatArray) -> FloatArray:
        """Compute LAC or stable, deterministic APS scores for every candidate."""
        if self.method == "lac":
            return 1 - proba
        if self.method != "aps":
            raise ValueError("method must be 'lac' or 'aps'")
        order = np.argsort(-proba, axis=1, kind="stable")
        cumulative = np.cumsum(np.take_along_axis(proba, order, axis=1), axis=1)
        scores = np.empty_like(proba)
        np.put_along_axis(scores, order, cumulative, axis=1)
        return scores

    def calibrate(self, X: FeatureMatrix, y: ArrayLike) -> Self:
        """Replace held-out scores and return this predictor for method chaining.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y
            One target per sample; classification labels follow the fitted classes.
        """
        X, y = self._validate_calibration(X, y)
        matches = y[:, None] == self.classes_[None, :]
        if not matches.any(axis=1).all():
            raise ValueError(
                "calibration targets contain labels absent from fitted classes_"
            )
        scores = self._candidate_scores(self._probabilities(X))
        self._store_scores(scores[np.arange(len(y)), matches.argmax(axis=1)])
        self.method_ = self.method
        return self

    def _calibrated_threshold(self, alpha: float) -> float:
        """Require an unchanged calibration method before selecting a threshold."""
        threshold = self._threshold(alpha)
        if self.method != self.method_:
            raise ValueError(
                "method changed after calibration; recalibrate before prediction"
            )
        return threshold

    def predict_point(self, X: FeatureMatrix) -> NDArray[Any]:
        """Return the wrapped estimator's labels without calibration.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        """
        X = self._validate_X(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return validated probabilities in classes_ order.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        """
        return self._probabilities(self._validate_X(X))

    def predict_set(self, X: FeatureMatrix, alpha: float = 0.05) -> NDArray[np.bool_]:
        """Return a boolean membership matrix of shape (n_samples, n_classes).

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        alpha
            Miscoverage probability strictly between zero and one.
        """
        X = self._validate_X(X)
        threshold = self._calibrated_threshold(alpha)
        return self._candidate_scores(self._probabilities(X)) <= threshold

    def predict_p_values(self, X: FeatureMatrix) -> FloatArray:
        """Return conservative conformal p-values, including calibration ties.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        """
        X = self._validate_X(X)
        if not self.is_calibrated_:
            raise RuntimeError("The estimator must be calibrated")
        if self.method != self.method_:
            raise ValueError("method changed after calibration; recalibrate")
        scores = self._candidate_scores(self._probabilities(X))
        return (
            self.n_calibration_
            + 1
            - np.searchsorted(self.sorted_calibration_scores_, scores, side="left")
        ) / (self.n_calibration_ + 1)

    def predict(
        self,
        X: FeatureMatrix,
        alpha: float = 0.05,
        *,
        return_set: Optional[bool] = None,
    ) -> Union[NDArray[Any], tuple[NDArray[Any], FloatArray]]:
        """Return (label sets with NaN exclusions, probabilities), or point labels.

                ``return_set=True`` forces the conformal tuple; ``False`` forces labels.
                This keyword can be forwarded by an outer sklearn Pipeline with
                metadata routing disabled. Numeric classes retain a numeric legacy array; string classes use an
                object array. Prefer predict_set for a dtype-independent representation.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        alpha
            Miscoverage probability strictly between zero and one.
        return_set
            Override the default output mode for this prediction call.
        """
        if return_set is not None and not isinstance(return_set, (bool, np.bool_)):
            raise ValueError("return_set must be boolean or None")
        if return_set is False or (
            return_set is None and self.prediction_mode == "point"
        ):
            return self.predict_point(X)
        X = self._validate_X(X)
        threshold = self._calibrated_threshold(alpha)
        proba = self._probabilities(X)
        mask = self._candidate_scores(proba) <= threshold
        dtype = float if np.issubdtype(self.classes_.dtype, np.number) else object
        if np.issubdtype(self.classes_.dtype, np.integer) and any(
            abs(int(label)) > 2**53 for label in self.classes_
        ):
            dtype = object  # float NaN sentinels must not round large integer labels
        labels = np.full(mask.shape, np.nan, dtype=dtype)
        labels[mask] = np.broadcast_to(self.classes_, mask.shape)[mask]
        return labels, proba

    def score(
        self, X: FeatureMatrix, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> float:
        """Return point accuracy regardless of prediction_mode.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        y
            One target per sample; classification labels follow the fitted classes.
        sample_weight
            Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
        """
        return accuracy_score(y, self.predict_point(X), sample_weight=sample_weight)

    def evaluate(
        self,
        X: FeatureMatrix,
        y: ArrayLike,
        alpha: float = 0.05,
        *,
        sample_weight: Optional[ArrayLike] = None,
    ) -> dict[str, float]:
        """Return accuracy, log loss, coverage and prediction set diagnostics.

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
        from sklearn.metrics import log_loss

        from .metrics import (
            prediction_set_coverage,
            prediction_set_empty_rate,
            prediction_set_singleton_rate,
            prediction_set_size,
        )

        X = self._validate_X(X)
        threshold = self._calibrated_threshold(alpha)
        proba = self._probabilities(X)
        mask = self._candidate_scores(proba) <= threshold
        if len(self.classes_) == 1:
            if np.asarray(y).shape != (X.shape[0],) or not np.all(
                np.asarray(y) == self.classes_[0]
            ):
                raise ValueError("y contains targets absent from fitted classes_")
            loss = 0.0
        else:
            loss = log_loss(y, proba, labels=self.classes_, sample_weight=sample_weight)
        return {
            "accuracy": self.score(X, y, sample_weight),
            "log_loss": loss,
            "coverage": prediction_set_coverage(
                y, mask, classes=self.classes_, sample_weight=sample_weight
            ),
            "mean_size": prediction_set_size(mask, sample_weight=sample_weight),
            "empty_rate": prediction_set_empty_rate(mask, sample_weight=sample_weight),
            "singleton_rate": prediction_set_singleton_rate(
                mask, sample_weight=sample_weight
            ),
        }
