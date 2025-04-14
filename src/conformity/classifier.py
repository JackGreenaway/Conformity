import numpy as np
from src.conformity.base import BaseConformalPredictor
from sklearn.base import BaseEstimator
from numpy.typing import ArrayLike


class ConformalClassifier(BaseConformalPredictor):
    def __init__(self, estimator: BaseEstimator) -> None:
        super().__init__()

        self.estimator = estimator

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> None:
        # predict class probabilities
        y_prob = self.estimator.predict_proba(X)

        # select true value
        true_probs = y_prob[np.arange(y_prob.shape[0]), y]

        # conformity measure - hinge loss in this case
        self.calibration_non_conformity = 1 - true_probs
        self.n_calib = self.calibration_non_conformity.shape[0]

        return None

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> ArrayLike:
        # predict class probabilities
        y_prob = self.estimator.predict_proba(X)
        # conformity measure - hinge loss in this case
        non_conformity = 1 - y_prob
        # compute empirical conformity distribution
        conformity_score = (
            self.calibration_non_conformity.shape[0]
            - np.searchsorted(
                np.sort(self.calibration_non_conformity), non_conformity, side="right"
            )
            + 1
        ) / (self.n_calib + 1)
        # calculate q_level
        q_level = np.ceil((self.n_calib + 1) * (1 - alpha)) / self.n_calib
        # boolean prediction set
        boolean_set = conformity_score > (1 - q_level)
        # create prediction set
        pred_set = np.where(
            boolean_set, np.vstack([self.estimator.classes_] * X.shape[0]), np.nan
        )

        return (pred_set, boolean_set, y_prob, q_level)
