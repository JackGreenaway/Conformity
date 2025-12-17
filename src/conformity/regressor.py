import numpy as np
import warnings
from conformity.base import BaseConformalPredictor
from sklearn.base import RegressorMixin
from numpy.typing import ArrayLike
from typing_extensions import Self


class ConformalRegressor(BaseConformalPredictor):
    """
    Conformal regressor for constructing prediction intervals using conformal prediction.

    This class wraps a regression estimator and provides calibrated prediction intervals
    with guaranteed coverage under exchangeability.

    Parameters
    ----------
    estimator : RegressorMixin
        A regression estimator implementing the scikit-learn interface.
    """

    def __init__(self, estimator: RegressorMixin) -> None:
        super().__init__(estimator=estimator)  # type: ignore

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Calibrate the conformal regressor using the provided calibration data.

        Parameters
        ----------
        X : ArrayLike
            Calibration features.
        y : ArrayLike
            Calibration targets.

        Returns
        -------
        Self
            The instance of the conformal regressor.
        """
        if self.is_calibrated_:
            warnings.warn("Estimator is already calibrated")

        y_pred = self.estimator_.predict(X)

        self.calibration_non_conformity = np.abs(y - y_pred)
        self.n_calib = self.calibration_non_conformity.shape[0]

        self.is_calibrated_ = True

        return self

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> tuple:
        """
        Make predictions with prediction intervals.

        Parameters
        ----------
        X : ArrayLike
            Features for which to make predictions.
        alpha : float, optional
            Significance level for the prediction intervals (default is 0.05).

        Returns
        -------
        tuple
            A tuple containing the predicted values, prediction intervals, and quantile level.
        """
        if not self.is_calibrated_:
            raise RuntimeError("Estimator has not been calibrated")

        if alpha > 1:
            raise ValueError(f"Alpha cannot be >= 1. Received {alpha}")

        y_pred = self.estimator_.predict(X)  # type: ignore

        quantile = np.ceil((self.n_calib + 1) * (1 - alpha)) / self.n_calib

        # clip quantile if necessary
        if quantile < 0.0 or quantile > 1.0:
            clipped_quantile = np.clip(quantile, 0.0, 1.0)

            warnings.warn(
                f"Quantile value {quantile} was clipped to {clipped_quantile} to fit within [0, 1]. "
                "This may indicate a very small calibration set, extreme alpha, and/or extreme values."
            )

            quantile = clipped_quantile

        y_pred_q_level = np.quantile(
            self.calibration_non_conformity,
            quantile,
        )
        y_pred_lower = y_pred - y_pred_q_level
        y_pred_higher = y_pred + y_pred_q_level

        return y_pred, np.column_stack((y_pred_lower, y_pred_higher)), y_pred_q_level
