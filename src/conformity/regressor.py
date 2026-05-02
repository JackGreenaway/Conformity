import numpy as np
import warnings
from conformity.base import BaseConformalPredictor
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from numpy.typing import ArrayLike
from typing_extensions import Self
from typing import Tuple


class ConformalRegressor(BaseConformalPredictor, RegressorMixin):
    """
    Conformal regressor for constructing prediction intervals using conformal prediction.

    This class wraps a regression estimator and provides calibrated prediction intervals
    with guaranteed marginal coverage under the exchangeability assumption. It is compatible
    with scikit-learn's ecosystem tools including pipelines and cross-validation.

    Parameters
    ----------
    estimator : RegressorMixin
        A regression estimator implementing the scikit-learn interface with
        `fit` and `predict` methods.

    Attributes
    ----------
    estimator_ : RegressorMixin
        The fitted base estimator (set after calling `fit`).
    is_calibrated_ : bool
        Whether the regressor has been calibrated. Initially False.
    calibration_non_conformity : ndarray of shape (n_calib,)
        Non-conformity scores (absolute residuals) computed on the calibration set.
    n_calib : int
        Number of calibration samples.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=100, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    >>> reg = ConformalRegressor(LinearRegression())
    >>> reg.fit(X_train, y_train)
    >>> reg.calibrate(X_calib, y_calib)
    >>> y_pred, intervals, q_level = reg.predict(X_test, alpha=0.1)
    """

    _estimator_type = "regressor"

    def __init__(self, estimator: RegressorMixin) -> None:
        """
        Initialize the conformal regressor.

        Parameters
        ----------
        estimator : RegressorMixin
            A regression estimator implementing the scikit-learn interface.
        """
        super().__init__(estimator=estimator)  # type: ignore

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Calibrate the conformal regressor using the provided calibration data.

        Parameters
        ----------
        X : array-like of shape (n_calib_samples, n_features)
            Calibration features.
        y : array-like of shape (n_calib_samples,)
            Calibration targets.

        Returns
        -------
        self : ConformalRegressor
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

        y_pred = self.estimator_.predict(X)
        self.calibration_non_conformity = np.abs(y - y_pred)
        self.n_calib = self.calibration_non_conformity.shape[0]
        self.is_calibrated_ = True

        return self

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Make predictions with prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for which to make predictions.
        alpha : float, default=0.05
            Significance level for the prediction intervals. Controls the coverage guarantee.
            Should be in the range (0, 1).

        Returns
        -------
        tuple of (ndarray, ndarray, float)
            - y_pred : ndarray of shape (n_samples,)
                Point predictions from the base estimator.
            - intervals : ndarray of shape (n_samples, 2)
                Prediction intervals with columns [lower, upper].
            - q_level : float
                The quantile level used for computing the prediction intervals.

        Raises
        ------
        RuntimeError
            If the regressor has not been calibrated before prediction.
        ValueError
            If alpha is not in the range (0, 1).
        """
        check_is_fitted(self, "estimator_")
        X = check_array(X, accept_sparse=False)

        if not self.is_calibrated_:
            raise RuntimeError(
                "The estimator must be calibrated before making predictions. "
                "Call the calibrate method with calibration data."
            )

        if alpha <= 0 or alpha >= 1:
            raise ValueError(
                f"alpha must be in the range (0, 1), got {alpha}"
            )

        y_pred = self.estimator_.predict(X)  # type: ignore

        quantile = np.ceil((self.n_calib + 1) * (1 - alpha)) / self.n_calib

        # clip quantile if necessary
        if quantile < 0.0 or quantile > 1.0:
            clipped_quantile = np.clip(quantile, 0.0, 1.0)
            warnings.warn(
                f"Quantile value {quantile:.4f} was clipped to {clipped_quantile:.4f} to fit within [0, 1]. "
                "This may indicate a very small calibration set, extreme alpha, and/or extreme values.",
                UserWarning
            )
            quantile = clipped_quantile

        y_pred_q_level = np.quantile(
            self.calibration_non_conformity,
            quantile,
        )
        y_pred_lower = y_pred - y_pred_q_level
        y_pred_higher = y_pred + y_pred_q_level

        return y_pred, np.column_stack((y_pred_lower, y_pred_higher)), y_pred_q_level
