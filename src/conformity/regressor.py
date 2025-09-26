import numpy as np
from src.conformity.base import BaseConformalPredictor
from sklearn.base import RegressorMixin
from numpy.typing import ArrayLike
from typing import Self


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
        super().__init__()

        self.estimator = estimator

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Calibrate the conformal regressor using calibration data.

        Parameters
        ----------
        X : ArrayLike
            Calibration feature matrix.
        y : ArrayLike
            Calibration target values.

        Returns
        -------
        Self
            The fitted conformal regressor.
        """
        y_pred = self.estimator.predict(X)  # type: ignore[attr-defined]

        self.calibration_non_conformity = np.abs(y - y_pred)
        self.n_calib = self.calibration_non_conformity.shape[0]

        return self

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> ArrayLike:
        """
        Predict using the conformal regressor and return prediction intervals.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix for which to predict.
        alpha : float, default=0.05
            Miscoverage rate (1 - confidence level).

        Returns
        -------
        tuple
            y_pred : ArrayLike
                Point predictions.
            prediction_intervals : ArrayLike
                Prediction intervals for each sample, shape (n_samples, 2).
            q_level : float
                Quantile level used for interval construction.
        """
        y_pred = self.estimator.predict(X)  # type: ignore[attr-defined]

        y_pred_q_level = np.quantile(
            self.calibration_non_conformity,
            np.ceil((self.n_calib + 1) * (1 - alpha)) / self.n_calib,
        )
        y_pred_lower = y_pred - y_pred_q_level
        y_pred_higher = y_pred + y_pred_q_level

        return y_pred, np.column_stack((y_pred_lower, y_pred_higher)), y_pred_q_level
