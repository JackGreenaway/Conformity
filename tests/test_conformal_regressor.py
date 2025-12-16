import numpy as np
import pytest
import warnings

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from conformity.regressor import ConformalRegressor


@pytest.fixture
def synthetic_regression_data():
    X, y = make_regression(
        n_samples=10_000, n_features=4, n_informative=3, random_state=927
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_fit_and_calibrate(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    assert reg.is_calibrated_
    assert hasattr(reg, "calibration_non_conformity")
    assert hasattr(reg, "n_calib")


def test_predict_interval_shape(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    y_pred, intervals, q_level = reg.predict(X_test, alpha=0.1)

    assert y_pred.shape[0] == X_test.shape[0]
    assert intervals.shape == (X_test.shape[0], 2)
    assert isinstance(q_level, float)


def test_predict_without_calibration_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)

    with pytest.raises(RuntimeError):
        reg.predict(X_test)


def test_multiple_calibrations_warn(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    with warnings.catch_warnings(record=True) as w:
        reg.calibrate(X_test, y_test)

        assert any("already calibrated" in str(warn.message) for warn in w)


def test_predict_with_different_alpha(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    _, intervals_01, _ = reg.predict(X_test, alpha=0.01)
    _, intervals_20, _ = reg.predict(X_test, alpha=0.20)

    assert np.all(
        (intervals_01[:, 1] - intervals_01[:, 0])
        >= (intervals_20[:, 1] - intervals_20[:, 0])
    )


def test_auto_calibrate_argument(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    reg.fit(
        np.concatenate([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        auto_calibrate=True,
        tts_kwargs={"test_size": 0.2, "random_state": 1},
    )

    assert reg.is_calibrated_


def test_interval_contains_true_value(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    y_pred, intervals, _ = reg.predict(X_test, alpha=0.1)

    # check that at least 85% of true values are within the intervals (since alpha=0.1, expect ~90%)
    coverage = np.mean((y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1]))

    assert coverage > 0.85


def test_extreme_inputs():
    X = np.array([[1e10], [-1e10], [0]])
    y = np.array([1e10, -1e10, 0])

    reg = ConformalRegressor(LinearRegression())

    reg.fit(X, y)
    reg.calibrate(X, y)

    with warnings.catch_warnings(record=True) as w:
        y_pred, intervals, _ = reg.predict(X)

        assert any("Quantile value" in str(warn.message) for warn in w)

    assert np.allclose(y_pred, y)
    assert np.all(intervals[:, 0] <= y_pred)
    assert np.all(intervals[:, 1] >= y_pred)
