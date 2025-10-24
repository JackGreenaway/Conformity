import numpy as np
import warnings
from src.conformity.base import BaseConformalPredictor
from sklearn.base import RegressorMixin
from numpy.typing import ArrayLike
from typing import Self


class ConformalRegressor(BaseConformalPredictor):
    def __init__(self, estimator: RegressorMixin) -> None:
        super().__init__(estimator=estimator)  # type: ignore

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        if self.is_calibrated_:
            warnings.warn("Estimator is already calibrated")

        y_pred = self.estimator_.predict(X)  # type: ignore[attr-defined]

        self.calibration_non_conformity = np.abs(y - y_pred)
        self.n_calib = self.calibration_non_conformity.shape[0]

        self.is_calibrated_ = True

        return self

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> tuple:
        if not self.is_calibrated_:
            raise RuntimeError("Estimator has not been calibrated")

        y_pred = self.estimator_.predict(X)  # type: ignore[attr-defined]

        y_pred_q_level = np.quantile(
            self.calibration_non_conformity,
            np.ceil((self.n_calib + 1) * (1 - alpha)) / self.n_calib,
        )
        y_pred_lower = y_pred - y_pred_q_level
        y_pred_higher = y_pred + y_pred_q_level

        return y_pred, np.column_stack((y_pred_lower, y_pred_higher)), y_pred_q_level
