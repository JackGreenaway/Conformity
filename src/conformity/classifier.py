import numpy as np
import warnings
from conformity.base import BaseConformalPredictor
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from numpy.typing import ArrayLike
from typing_extensions import Self
from typing import Tuple


class ConformalClassifier(BaseConformalPredictor, ClassifierMixin):
    """
    Conformal classifier for constructing prediction sets using conformal prediction.

    This class wraps a classification estimator and provides calibrated prediction sets
    with guaranteed marginal coverage under exchangeability assumption. It is compatible
    with scikit-learn's ecosystem tools including pipelines and cross-validation.

    Parameters
    ----------
    estimator : ClassifierMixin
        A classification estimator implementing the scikit-learn interface with
        `fit`, `predict`, and `predict_proba` methods.

    Attributes
    ----------
    estimator_ : ClassifierMixin
        The fitted base estimator (set after calling `fit`).
    is_calibrated_ : bool
        Whether the classifier has been calibrated. Initially False.
    calibration_non_conformity : ndarray of shape (n_calib,)
        Non-conformity scores computed on the calibration set.
    n_calib : int
        Number of calibration samples.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    >>> clf = ConformalClassifier(LogisticRegression(max_iter=1000))
    >>> clf.fit(X_train, y_train)
    >>> clf.calibrate(X_calib, y_calib)
    >>> pred_set, _, _, _ = clf.predict(X_test, alpha=0.1)
    """

    _estimator_type = "classifier"

    def __init__(self, estimator: ClassifierMixin) -> None:
        """
        Initialize the conformal classifier.

        Parameters
        ----------
        estimator : ClassifierMixin
            A classification estimator implementing the scikit-learn interface.
        """
        super().__init__(estimator=estimator)  # type: ignore

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Calibrate the conformal classifier using the provided calibration data.

        Parameters
        ----------
        X : array-like of shape (n_calib_samples, n_features)
            Calibration features.
        y : array-like of shape (n_calib_samples,)
            Calibration targets.

        Returns
        -------
        self : ConformalClassifier
            Calibrated estimator.
        """
        check_is_fitted(self, "estimator_")
        X = check_array(X, accept_sparse=False)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {X.shape[0]} != {y.shape[0]}"
            )

        if self.is_calibrated_:
            warnings.warn(
                "The estimator is already calibrated. Recalibrating may affect prediction quality.",
                UserWarning
            )

        y_prob = self.estimator_.predict_proba(X)
        true_probs = y_prob[np.arange(y_prob.shape[0]), y.astype(int)]
        self.calibration_non_conformity = 1 - true_probs
        self.n_calib = self.calibration_non_conformity.shape[0]
        self.is_calibrated_ = True

        return self

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> Tuple:
        """
        Make predictions with prediction sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for which to make predictions.
        alpha : float, default=0.05
            Significance level for the prediction sets. Controls the coverage guarantee.
            Recommended range is (0, 1).

        Returns
        -------
        tuple of (ndarray, ndarray, ndarray, float)
            - pred_set : ndarray of shape (n_samples, n_classes)
                Prediction sets for each sample. NaN indicates classes not in the set.
            - boolean_set : ndarray of shape (n_samples, n_classes)
                Boolean mask indicating membership in prediction set.
            - y_prob : ndarray of shape (n_samples, n_classes)
                Class probabilities from the base estimator.
            - q_level : float
                The quantile level used for computing the prediction sets.

        Raises
        ------
        RuntimeError
            If the classifier has not been calibrated before prediction.
        """
        check_is_fitted(self, "estimator_")
        X = check_array(X, accept_sparse=False)

        if not self.is_calibrated_:
            raise RuntimeError(
                "The estimator must be calibrated before making predictions. "
                "Call the calibrate method with calibration data."
            )

        y_prob = self.estimator_.predict_proba(X)
        non_conformity = 1 - y_prob

        conformity_score = (
            self.calibration_non_conformity.shape[0]
            - np.searchsorted(
                np.sort(self.calibration_non_conformity), non_conformity, side="right"
            )
            + 1
        ) / (self.n_calib + 1)

        q_level = np.ceil((self.n_calib + 1) * (1 - alpha)) / self.n_calib

        boolean_set = conformity_score > (1 - q_level)
        pred_set = np.where(
            boolean_set,
            np.vstack([self.estimator_.classes_] * X.shape[0]),  # type: ignore[attr-defined]
            np.nan,
        )

        return (pred_set, boolean_set, y_prob, q_level)
