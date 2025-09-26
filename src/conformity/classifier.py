import numpy as np
from src.conformity.base import BaseConformalPredictor
from sklearn.base import ClassifierMixin
from numpy.typing import ArrayLike
from typing import Self


class ConformalClassifier(BaseConformalPredictor):
    """
    Conformal classifier for constructing prediction sets using conformal prediction.

    This class wraps a classification estimator and provides calibrated prediction sets
    with guaranteed coverage under exchangeability.

    Parameters
    ----------
    estimator : ClassifierMixin
        A classification estimator implementing the scikit-learn interface.
    """

    def __init__(self, estimator: ClassifierMixin) -> None:
        super().__init__()

        self.estimator = estimator

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Calibrate the conformal classifier using calibration data.

        Parameters
        ----------
        X : ArrayLike
            Calibration feature matrix.
        y : ArrayLike
            Calibration target values.

        Returns
        -------
        Self
            The fitted conformal classifier.
        """
        y_prob = self.estimator.predict_proba(X)  # type: ignore[attr-defined]

        true_probs = y_prob[np.arange(y_prob.shape[0]), y]

        self.calibration_non_conformity = 1 - true_probs
        self.n_calib = self.calibration_non_conformity.shape[0]

        return self

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> ArrayLike:
        """
        Predict using the conformal classifier and return prediction sets.

        Parameters
        ----------
        X : ArrayLike
            Feature matrix for which to predict.
        alpha : float, default=0.05
            Miscoverage rate (1 - confidence level).

        Returns
        -------
        tuple
            pred_set : ArrayLike
                Prediction sets for each sample.
            boolean_set : ArrayLike
                Boolean mask indicating which classes are included in the prediction set.
            y_prob : ArrayLike
                Predicted class probabilities for each sample.
            q_level : float
                Quantile level used for set construction.
        """
        y_prob = self.estimator.predict_proba(X)  # type: ignore[attr-defined]

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
            np.vstack([self.estimator.classes_] * X.shape[0]),  # type: ignore[attr-defined]
            np.nan,
        )

        return (pred_set, boolean_set, y_prob, q_level)
