from sklearn.base import BaseEstimator
from numpy.typing import ArrayLike


class BaseConformalPredictor:
    """
    Base class for conformal predictors.

    Provides a common interface for conformal regression and classification estimators.
    """

    def __init__(self):
        """
        Initialise the base conformal predictor.
        """
        pass

    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
        """
        Fit the underlying estimator to the training data.

        Parameters
        ----------
        X : ArrayLike
            Training feature matrix.
        y : ArrayLike
            Training target values.

        Returns
        -------
        BaseEstimator
            The fitted estimator.
        """
        self.estimator = self.estimator_.fit(X, y)  # type: ignore[attr-defined]

        return self.estimator

    @property
    def estimator_(self) -> BaseEstimator:
        """
        Access the fitted estimator.

        Returns
        -------
        BaseEstimator
            The fitted estimator.
        """
        return self.estimator
