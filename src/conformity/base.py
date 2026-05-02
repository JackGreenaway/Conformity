from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from typing_extensions import Self
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
from typing import Dict, Any, Optional


class BaseConformalPredictor(BaseEstimator, ABC):
    """
    Abstract base class for conformal predictors.

    This class provides the interface for conformal predictors, which can be used
    for both regression and classification tasks. It ensures compatibility with
    scikit-learn conventions and ecosystem tools.

    Parameters
    ----------
    estimator : BaseEstimator
        The base estimator to be wrapped by the conformal predictor. Must implement
        `fit` and `predict` methods following scikit-learn conventions.

    Attributes
    ----------
    estimator_ : BaseEstimator
        The fitted base estimator (set after calling `fit`).
    is_calibrated_ : bool
        Whether the predictor has been calibrated. Initialized to False.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        """
        Initialize the conformal predictor with a base estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator to be used for conformal prediction.
        """
        self.estimator = estimator
        self.is_calibrated_ = False

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        auto_calibrate: bool = False,
        tts_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Self:
        """
        Fit the conformal predictor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        auto_calibrate : bool, default=False
            Whether to automatically calibrate the model after fitting. If True,
            the training data will be split into training and calibration sets.
        tts_kwargs : dict or None, default=None
            Optional keyword arguments passed to
            `sklearn.model_selection.train_test_split` when `auto_calibrate=True`.
            Default split is 80/20 if not specified.

        Returns
        -------
        self : BaseConformalPredictor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)

        if auto_calibrate:
            if tts_kwargs is None:
                tts_kwargs = {"test_size": 0.2, "random_state": None}

            X_train, X_calib, y_train, y_calib = train_test_split(X, y, **tts_kwargs)
            self.estimator_ = clone(self.estimator).fit(X_train, y_train)
            self.calibrate(X=X_calib, y=y_calib)

        else:
            self.estimator_ = clone(self.estimator).fit(X, y)

        return self

    @abstractmethod
    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Calibrate the conformal predictor using calibration data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration features.
        y : array-like of shape (n_samples,)
            Calibration targets.

        Returns
        -------
        self : BaseConformalPredictor
            Calibrated estimator.
        """
        pass
