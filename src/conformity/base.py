from sklearn.base import BaseEstimator, clone
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
from typing import Self


class BaseConformalPredictor(ABC):
    def __init__(self, estimator: BaseEstimator) -> None:
        self.base_estimator = estimator
        self.is_calibrated_ = False

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        auto_calibrate: bool = False,
        tts_kwargs: dict = {},
    ) -> Self:
        if auto_calibrate:
            X_train, X_calib, y_train, y_calib = train_test_split(X, y, **tts_kwargs)

            self.estimator_ = clone(self.base_estimator).fit(X_train, y_train)
            self.calibrate(X=X_calib, y=y_calib)

            self.is_calibrated_ = True

        else:
            self.estimator_ = clone(self.base_estimator).fit(X, y)

        return self

    @abstractmethod
    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        # implement model calibration
        pass
