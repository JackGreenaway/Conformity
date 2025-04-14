import numpy as np
from src.conformity.base import BaseConformalPredictor
from sklearn.base import BaseEstimator
from numpy.typing import ArrayLike


class ConformalRegressor(BaseConformalPredictor):
    def __init__(self, estimator: BaseEstimator) -> None:
        super().__init__()

        self.estimator = estimator

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> None:
        # predict class probabilities
        y_pred = self.estimator.predict(X)

        # conformity measure - hinge loss in this case
        self.calibration_non_conformity = np.abs(y - y_pred)
        self.n_calib = self.calibration_non_conformity.shape[0]

        return None

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> ArrayLike:
        # predict class probabilities
        y_pred = self.estimator.predict(X)
        # compute empirical conformity distribution
        y_pred_q_level = np.quantile(
            self.calibration_non_conformity,
            np.ceil((self.n_calib + 1) * (1 - alpha)) / self.n_calib,
        )
        # create prediction intervals
        y_pred_lower = y_pred - y_pred_q_level
        y_pred_higher = y_pred + y_pred_q_level

        return y_pred, np.column_stack((y_pred_lower, y_pred_higher)), y_pred_q_level
