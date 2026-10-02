"""Label-safe least ambiguous and adaptive prediction sets."""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from .base import BaseConformalPredictor


class ConformalClassifier(ClassifierMixin, BaseConformalPredictor):
    """Split conformal classification using LAC or deterministic APS scores.

    ``method='lac'`` scores 1 - probability; ``'aps'`` scores cumulative
    descending probability through the candidate class. Ties use stable class
    order. ``predict_set`` returns a boolean matrix in ``classes_`` order, which
    works with any sklearn class labels. Empty prediction sets are permitted.
    """

    def __init__(
        self,
        estimator,
        *,
        method="lac",
        auto_calibrate=False,
        tts_kwargs=None,
        calibration_size=0.2,
        random_state=None,
        prediction_mode="conformal",
    ):
        super().__init__(
            estimator,
            auto_calibrate=auto_calibrate,
            tts_kwargs=tts_kwargs,
            calibration_size=calibration_size,
            random_state=random_state,
            prediction_mode=prediction_mode,
        )
        self.method = method

    def _validate_estimator(self, estimator):
        super()._validate_estimator(estimator)
        if self.method not in ("lac", "aps"):
            raise ValueError("method must be 'lac' or 'aps'")
        if not callable(getattr(estimator, "predict_proba", None)):
            raise TypeError("classification estimator must implement predict_proba")

    def _probabilities(self, X):
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

    def _candidate_scores(self, proba):
        if self.method == "lac":
            return 1 - proba
        if self.method != "aps":
            raise ValueError("method must be 'lac' or 'aps'")
        order = np.argsort(-proba, axis=1, kind="stable")
        cumulative = np.cumsum(np.take_along_axis(proba, order, axis=1), axis=1)
        scores = np.empty_like(proba)
        np.put_along_axis(scores, order, cumulative, axis=1)
        return scores

    def calibrate(self, X, y):
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

    def _calibrated_threshold(self, alpha):
        threshold = self._threshold(alpha)
        if self.method != self.method_:
            raise ValueError(
                "method changed after calibration; recalibrate before prediction"
            )
        return threshold

    def predict_point(self, X):
        """Return the wrapped estimator's labels without calibration."""
        X = self._validate_X(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Return validated probabilities in classes_ order."""
        return self._probabilities(self._validate_X(X))

    def predict_set(self, X, alpha=0.05):
        """Return a boolean membership matrix of shape (n_samples, n_classes)."""
        X = self._validate_X(X)
        threshold = self._calibrated_threshold(alpha)
        return self._candidate_scores(self._probabilities(X)) <= threshold

    def predict_p_values(self, X):
        """Return conservative conformal p-values, including calibration ties."""
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

    def predict(self, X, alpha=0.05, *, return_set=None):
        """Return (label sets with NaN exclusions, probabilities), or point labels.

        ``return_set=True`` forces the conformal tuple; ``False`` forces labels.
        This keyword can be forwarded by an outer sklearn Pipeline with
        metadata routing disabled. Numeric classes retain a numeric legacy array; string classes use an
        object array. Prefer predict_set for a dtype-independent representation.
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

    def score(self, X, y, sample_weight=None):
        """Return point accuracy regardless of prediction_mode."""
        return accuracy_score(y, self.predict_point(X), sample_weight=sample_weight)

    def evaluate(self, X, y, alpha=0.05, *, sample_weight=None):
        """Return accuracy, log loss, coverage and prediction set diagnostics."""
        from sklearn.metrics import log_loss
        from .metrics import (
            prediction_set_coverage,
            prediction_set_size,
            prediction_set_empty_rate,
            prediction_set_singleton_rate,
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
