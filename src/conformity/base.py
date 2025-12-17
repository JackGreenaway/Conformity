from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from typing_extensions import Self
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from typing import Dict, Any


class BaseConformalPredictor(ABC):
    """
    Abstract base class for conformal predictors.

    This class provides the interface for conformal predictors, which can be used
    for both regression and classification tasks.

    Parameters
    ----------
    estimator : BaseEstimator
        The base estimator to be wrapped by the conformal predictor.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        """
        Initialise the conformal predictor with a base estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be used for conformal prediction.
        """
        self.base_estimator = estimator
        self.is_calibrated_ = False

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        auto_calibrate: bool = False,
        tts_kwargs: Dict[str, Any] | None = None,
    ) -> Self:
        """
        Fit the conformal predictor to the training data.

        Parameters
        ----------
        X : ArrayLike
            Training features.
        y : ArrayLike
            Training targets.
        auto_calibrate : bool, optional
            Whether to automatically calibrate the model after fitting (default is False).
        tts_kwargs : dict or None, optional
            Optional kwargs passed to `sklearn.model_selection.train_test_split` when
            auto_calibrate=True.

        Returns
        -------
        Self
            The instance of the conformal predictor.
        """
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
