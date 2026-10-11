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

    Attributes
    ----------
    method_ : {"lac", "aps"}
        Score method recorded by successful calibration; changing it requires
        recalibration. Shared fitted attributes are documented on the base class.
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from conformity import ConformalClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> X_dev, X_test, y_dev, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=0
    ... )
    >>> X_train, X_calib, y_train, y_calib = train_test_split(
    ...     X_dev, y_dev, test_size=0.25, random_state=1
    ... )
    >>> model = Pipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("conformal", ConformalClassifier(
    ...         LogisticRegression(max_iter=500), method="lac", auto_calibrate=False
    ...     )),
    ... ])
    >>> _ = model.fit(X_train, y_train)
    >>> _ = model[-1].calibrate(model[:-1].transform(X_calib), y_calib)
    >>> mask, probabilities = model.predict(X_test, alpha=0.1, return_set=True)
    >>> mask.shape, probabilities.shape
    ((30, 3), (30, 3))
    >>> model.classes_.tolist()
    [0, 1, 2]
    >>> mask.dtype.name
    'bool'
    """

    # Score method recorded when calibration succeeds.
    method_: Literal["lac", "aps"]

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        method: Literal["lac", "aps"] = "lac",
        auto_calibrate: bool = True,
        tts_kwargs: Optional[dict[str, Any]] = None,
        calibration_size: Union[float, int] = 0.2,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        """Configure the wrapped estimator and held-out calibration workflow.

        Parameters
        ----------
        estimator : BaseEstimator
            Cloneable sklearn classifier implementing predict_proba.
        method : {'lac', 'aps'}
            Use lac (1 - probability) or aps (cumulative ranked probability).
        auto_calibrate : bool or None
            Reserve held-out calibration data during fit; defaults to True.
        tts_kwargs : dict or None
            Optional train_test_split overrides, applied only with auto_calibrate.
        calibration_size : float or int
            Calibration fraction or sample count; defaults to 0.2.
        random_state : int, RandomState or None
            Seed or RandomState for reproducible automatic splits.
        """
        super().__init__(
            estimator,
            auto_calibrate=auto_calibrate,
            tts_kwargs=tts_kwargs,
            calibration_size=calibration_size,
            random_state=random_state,
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
        X = self._validate_X(X)
        return self.estimator_.predict(X)

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return validated probabilities in classes_ order.

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Finite class probabilities in fitted classes_ order.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.

        """
        return self._probabilities(self._validate_X(X))

    def predict_set(self, X: FeatureMatrix, alpha: float = 0.05) -> NDArray[np.bool_]:
        """Return a boolean membership matrix of shape (n_samples, n_classes).

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        alpha : float
            Miscoverage probability strictly between zero and one.

        Returns
        -------
        membership : ndarray of bool, shape (n_samples, n_classes)
            Included labels in classes_ order; empty and all-class sets are permitted.

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
        threshold = self._calibrated_threshold(alpha)
        return self._candidate_scores(self._probabilities(X)) <= threshold

    def predict_p_values(self, X: FeatureMatrix) -> FloatArray:
        """Return conservative conformal p-values, including calibration ties.

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.

        Returns
        -------
        p_values : ndarray of shape (n_samples, n_classes)
            Conservative conformal p-values in classes_ order, including ties.

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
        return_set: bool = False,
    ) -> Union[NDArray[Any], tuple[NDArray[Any], FloatArray]]:
        """Return point labels, or (boolean membership sets, probabilities).

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Feature matrix of shape (n_samples, n_features); dense, sparse, or DataFrame.
        alpha : float
            Miscoverage probability strictly between zero and one.
        return_set : bool
            Request conformal output alongside ordinary predictions.

        Returns
        -------
        predictions : ndarray or tuple of ndarray
            Point labels, or (membership, probabilities) when return_set=True.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.
        RuntimeError
            If conformal output is requested before calibration.

        """
        if not isinstance(return_set, (bool, np.bool_)):
            raise ValueError("return_set must be boolean")
        if not return_set:
            return self.predict_point(X)
        X = self._validate_X(X)
        threshold = self._calibrated_threshold(alpha)
        proba = self._probabilities(X)
        mask = self._candidate_scores(proba) <= threshold
        return mask, proba

    def score(
        self, X: FeatureMatrix, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> float:
        """Return point accuracy using point predictions.

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
            Keys are accuracy, log_loss, coverage, mean_size, empty_rate and singleton_rate.

        Raises
        ------
        ValueError
            If input shapes, values, weights or configuration are invalid.
        NotFittedError
            If fit has not established a fitted estimator.
        RuntimeError
            If conformal output is requested before calibration.

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
