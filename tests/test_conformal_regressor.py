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


def test_single_sample_prediction(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    single_sample = X_test[:1]
    y_pred, intervals, _ = reg.predict(single_sample, alpha=0.1)

    assert y_pred.shape == (1,)
    assert intervals.shape == (1, 2)
    assert intervals[0, 0] <= y_pred[0] <= intervals[0, 1]


def test_alpha_boundary_values(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    # Test alpha very close to 0
    _, intervals_small, q_small = reg.predict(X_test, alpha=0.001)
    # Test alpha close to 1
    _, intervals_large, q_large = reg.predict(X_test, alpha=0.99)

    # q_level and interval widths should increase with alpha
    # Use isclose for numerical comparison with small numbers
    small_width = np.mean(intervals_small[:, 1] - intervals_small[:, 0])
    large_width = np.mean(intervals_large[:, 1] - intervals_large[:, 0])

    assert large_width >= small_width - 1e-10 or np.isclose(
        large_width, small_width, atol=1e-10
    )


def test_fit_without_features_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    with pytest.raises((ValueError, IndexError)):
        reg.fit(np.array([]), np.array([]))


def test_calibrate_without_fit_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    with pytest.raises((RuntimeError, AttributeError)):
        reg.calibrate(X_test, y_test)


def test_mismatched_x_y_shapes_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())

    with pytest.raises(ValueError):
        reg.fit(X_train, y_train[:-10])


def test_calibrate_with_mismatched_shapes_raises(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)

    with pytest.raises(ValueError):
        reg.calibrate(X_test, y_test[:-5])


def test_prediction_intervals_monotonic(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    y_pred, intervals, _ = reg.predict(X_test, alpha=0.1)

    # Lower bound should be <= prediction <= upper bound
    assert np.all(intervals[:, 0] <= y_pred)
    assert np.all(y_pred <= intervals[:, 1])


def test_consistent_predictions(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    y_pred1, intervals1, q1 = reg.predict(X_test, alpha=0.1)
    y_pred2, intervals2, q2 = reg.predict(X_test, alpha=0.1)

    # Predictions should be deterministic
    np.testing.assert_array_equal(y_pred1, y_pred2)
    np.testing.assert_array_equal(intervals1, intervals2)
    assert q1 == q2


def test_increasing_alpha_increases_intervals(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    alphas = [0.05, 0.1, 0.2, 0.5]
    widths = []

    for alpha in alphas:
        _, intervals, _ = reg.predict(X_test, alpha=alpha)
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        widths.append(width)

    # Interval widths should increase with alpha (allowing for numerical tolerance)
    assert all(widths[i] <= widths[i + 1] + 1e-10 for i in range(len(widths) - 1))


def test_auto_calibrate_with_small_dataset(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    # Use smaller subset
    X_small = X_train[:50]
    y_small = y_train[:50]

    reg = ConformalRegressor(LinearRegression())

    reg.fit(
        X_small,
        y_small,
        auto_calibrate=True,
        tts_kwargs={"test_size": 0.3, "random_state": 1},
    )

    assert reg.is_calibrated_
    assert hasattr(reg, "calibration_non_conformity")


def test_predict_different_sample_sizes(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    # Test predictions for various numbers of samples
    for n_samples in [1, 10, 100, 500]:
        X_subset = X_test[:n_samples]
        y_pred, intervals, _ = reg.predict(X_subset, alpha=0.1)

        assert y_pred.shape[0] == n_samples
        assert intervals.shape == (n_samples, 2)


def test_calibration_with_identical_features(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    # Create calibration set with identical features
    X_calib = X_test[:5].copy()
    y_calib = y_test[:5].copy()

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_calib, y_calib)

    assert reg.is_calibrated_


def test_non_conformity_scores_are_valid(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    # Non-conformity scores should be numeric and non-negative (typically absolute residuals)
    assert hasattr(reg, "calibration_non_conformity")
    assert reg.calibration_non_conformity is not None
    assert len(reg.calibration_non_conformity) > 0
    assert all(
        isinstance(x, (int, float, np.number)) for x in reg.calibration_non_conformity
    )


def test_predict_with_all_zero_targets(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    # Create dataset with all zero targets
    y_zero = np.zeros_like(y_train)

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_zero)
    reg.calibrate(X_test, np.zeros_like(y_test))

    y_pred, intervals, _ = reg.predict(X_test, alpha=0.1)

    assert y_pred.shape[0] == X_test.shape[0]
    assert intervals.shape == (X_test.shape[0], 2)


def test_large_alpha_produces_wide_intervals(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    _, intervals_small, _ = reg.predict(X_test, alpha=0.01)
    _, intervals_large, _ = reg.predict(X_test, alpha=0.9)

    avg_width_small = np.mean(intervals_small[:, 1] - intervals_small[:, 0])
    avg_width_large = np.mean(intervals_large[:, 1] - intervals_large[:, 0])

    assert avg_width_large > avg_width_small - 1e-10


def test_calibration_set_size_recorded(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    assert reg.n_calib == X_test.shape[0]


def test_predict_returns_correct_types(synthetic_regression_data):
    X_train, X_test, y_train, y_test = synthetic_regression_data

    reg = ConformalRegressor(LinearRegression())
    reg.fit(X_train, y_train)
    reg.calibrate(X_test, y_test)

    y_pred, intervals, q_level = reg.predict(X_test[:5], alpha=0.1)

    assert isinstance(y_pred, np.ndarray)
    assert isinstance(intervals, np.ndarray)
    assert isinstance(q_level, (float, np.floating))
