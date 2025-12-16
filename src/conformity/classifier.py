import numpy as np
import warnings
from conformity.base import BaseConformalPredictor
from sklearn.base import ClassifierMixin
from numpy.typing import ArrayLike
from typing_extensions import Self


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
        super().__init__(estimator=estimator)  # type: ignore

        # self.estimator = estimator

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Calibrate the conformal classifier using the provided calibration data.

        Parameters
        ----------
        X : ArrayLike
            Calibration features.
        y : ArrayLike
            Calibration targets.

        Returns
        -------
        Self
            The instance of the conformal classifier.
        """
        if self.is_calibrated_:
            warnings.warn("Estimator is already calibrated")

        y_prob = self.estimator_.predict_proba(X)

        true_probs = y_prob[np.arange(y_prob.shape[0]), y]

        self.calibration_non_conformity = 1 - true_probs
        self.n_calib = self.calibration_non_conformity.shape[0]

        self.is_calibrated_ = True

        return self

    def predict(self, X: ArrayLike, alpha: float = 0.05) -> tuple:
        """
        Make predictions with prediction sets.

        Parameters
        ----------
        X : ArrayLike
            Features for which to make predictions.
        alpha : float, optional
            Significance level for the prediction sets (default is 0.05).

        Returns
        -------
        tuple
            A tuple containing the predicted sets, boolean indicators, predicted probabilities, and quantile level.
        """
        if not self.is_calibrated_:
            raise RuntimeError("Estimator has not been calibrated")

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
