from sklearn.base import BaseEstimator
from numpy.typing import ArrayLike


class BaseConformalPredictor:
    def __init__(self):
        pass

    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseEstimator:
        self.estimator = self.estimator_.fit(X, y)  # type: ignore[attr-defined]

        return self.estimator

    @property
    def estimator_(self) -> BaseEstimator:
        return self.estimator
