import numpy as np
import pytest
import warnings

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from conformity.base import BaseConformalPredictor


@pytest.fixture
def synthetic_regression_data():
    X, y = make_regression(
        n_samples=1_000, n_features=4, n_informative=3, random_state=927
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)


class ConcreteConformalPredictor(BaseConformalPredictor):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, estimator):
        super().__init__(estimator)
        self.estimator = estimator

    def _compute_non_conformity_scores(self, X, y):
        """Compute absolute residuals as non-conformity scores."""
        predictions = self.estimator.predict(X)
        return np.abs(y - predictions)

    def _make_prediction(self, X, q_level):
        """Return point predictions."""
        return self.estimator.predict(X)

    def fit(self, X, y, auto_calibrate=False, tts_kwargs=None):
        """Fit the estimator."""
        self.estimator.fit(X, y)

        if auto_calibrate:
            if tts_kwargs is None:
                tts_kwargs = {}
            X_train, X_calib, y_train, y_calib = train_test_split(X, y, **tts_kwargs)
            self.estimator.fit(X_train, y_train)
            self.calibrate(X_calib, y_calib)

        return self

    def calibrate(self, X, y):
        """Calibrate the conformal predictor."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if self.is_calibrated_:
            warnings.warn("Predictor is already calibrated. Recalibrating...")

        self.calibration_non_conformity = self._compute_non_conformity_scores(X, y)
        self.n_calib = X.shape[0]
        self.is_calibrated_ = True

        return self

    def predict(self, X, alpha=0.1):
        """Make predictions with conformal intervals."""
        if not self.is_calibrated_:
            raise RuntimeError("Predictor must be calibrated before making predictions")
        return self._make_prediction(X, alpha)


def test_base_fit(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    predictor.fit(X_train, y_train)

    assert hasattr(predictor, "estimator")
    assert predictor.estimator is not None


def test_base_calibrate(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    predictor.fit(X_train, y_train)
    predictor.calibrate(X_test, y_test)

    assert predictor.is_calibrated_
    assert hasattr(predictor, "calibration_non_conformity")
    assert hasattr(predictor, "n_calib")
    assert predictor.n_calib == X_test.shape[0]


def test_base_predict_without_calibration_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())
    predictor.fit(X_train, y_train)

    with pytest.raises(RuntimeError):
        predictor.predict(X_test)


def test_base_multiple_calibrations_warn(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    predictor.fit(X_train, y_train)
    predictor.calibrate(X_test, y_test)

    with warnings.catch_warnings(record=True) as w:
        predictor.calibrate(X_test, y_test)

        assert any("already calibrated" in str(warn.message) for warn in w)


def test_base_auto_calibrate(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    predictor.fit(
        np.concatenate([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        auto_calibrate=True,
        tts_kwargs={"test_size": 0.2, "random_state": 1},
    )

    assert predictor.is_calibrated_


def test_base_calibrate_without_fit_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    with pytest.raises((RuntimeError, AttributeError)):
        predictor.calibrate(X_test, y_test)


def test_base_mismatched_x_y_shapes_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    with pytest.raises(ValueError):
        predictor.fit(X_train, y_train[:-10])


def test_base_calibrate_with_mismatched_shapes_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())
    predictor.fit(X_train, y_train)

    with pytest.raises(ValueError):
        predictor.calibrate(X_test, y_test[:-5])


def test_base_non_conformity_scores_computed(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    predictor.fit(X_train, y_train)
    predictor.calibrate(X_test, y_test)

    assert hasattr(predictor, "calibration_non_conformity")
    assert len(predictor.calibration_non_conformity) == X_test.shape[0]
    assert all(x >= 0 for x in predictor.calibration_non_conformity)


def test_base_estimator_is_stored(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    estimator = LinearRegression()
    predictor = ConcreteConformalPredictor(estimator)

    predictor.fit(X_train, y_train)

    assert predictor.estimator is estimator


def test_base_calibration_flag(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    assert not predictor.is_calibrated_

    predictor.fit(X_train, y_train)
    assert not predictor.is_calibrated_

    predictor.calibrate(X_test, y_test)
    assert predictor.is_calibrated_


def test_base_empty_fit_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    with pytest.raises((ValueError, IndexError)):
        predictor.fit(np.array([]), np.array([]))


def test_base_calibration_n_calib_matches(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    predictor.fit(X_train, y_train)

    # Calibrate with subset
    X_calib = X_test[:50]
    y_calib = y_test[:50]

    predictor.calibrate(X_calib, y_calib)

    assert predictor.n_calib == 50


def test_base_different_estimators(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    from sklearn.tree import DecisionTreeRegressor

    for EstimatorClass in [LinearRegression, DecisionTreeRegressor]:
        predictor = ConcreteConformalPredictor(EstimatorClass())

        predictor.fit(X_train, y_train)
        predictor.calibrate(X_test, y_test)

        assert predictor.is_calibrated_


def test_base_fit_preserves_data_shapes(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())

    original_X_shape = X_train.shape
    original_y_shape = y_train.shape

    predictor.fit(X_train, y_train)

    # Ensure data wasn't modified in place
    assert X_train.shape == original_X_shape
    assert y_train.shape == original_y_shape


def test_base_calibrate_preserves_data_shapes(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())
    predictor.fit(X_train, y_train)

    original_X_shape = X_test.shape
    original_y_shape = y_test.shape

    predictor.calibrate(X_test, y_test)

    # Ensure data wasn't modified in place
    assert X_test.shape == original_X_shape
    assert y_test.shape == original_y_shape


def test_base_single_sample_calibration(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    predictor = ConcreteConformalPredictor(LinearRegression())
    predictor.fit(X_train, y_train)

    # Calibrate with single sample
    X_calib = X_test[:1]
    y_calib = y_test[:1]

    predictor.calibrate(X_calib, y_calib)

    assert predictor.is_calibrated_
    assert predictor.n_calib == 1
