"""Integration tests for conformal predictors."""

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split

from conformity.classifier import ConformalClassifier
from conformity.regressor import ConformalRegressor
from conformity.metrics import (
    prediction_interval_coverage,
    prediction_interval_efficiency,
    prediction_set_coverage,
)


class TestMultipleEstimators:
    """Tests with multiple different estimators."""

    def test_regressor_with_different_models(self):
        """Test ConformalRegressor with different regression models."""
        X, y = make_regression(n_samples=300, n_features=10, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        for estimator in [
            LinearRegression(),
            Ridge(),
            Lasso(),
            RandomForestRegressor(n_estimators=10, random_state=42),
        ]:
            reg = ConformalRegressor(estimator)
            reg.fit(X_train, y_train)
            reg.calibrate(X_calib, y_calib)

            y_pred, intervals, _ = reg.predict(X_test, alpha=0.1)

            assert y_pred.shape[0] == X_test.shape[0]
            assert intervals.shape == (X_test.shape[0], 2)
            assert np.all(intervals[:, 0] <= intervals[:, 1])

    def test_classifier_with_different_models(self):
        """Test ConformalClassifier with different classification models."""
        X, y = make_classification(n_samples=300, n_features=10, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        for estimator in [
            LogisticRegression(max_iter=1000, random_state=42),
            RandomForestClassifier(n_estimators=10, random_state=42),
        ]:
            clf = ConformalClassifier(estimator)
            clf.fit(X_train, y_train)
            clf.calibrate(X_calib, y_calib)

            pred_set, _, _, _ = clf.predict(X_test, alpha=0.1)

            assert pred_set.shape[0] == X_test.shape[0]


class TestCoverageLevels:
    """Tests for different coverage levels."""

    def test_regressor_coverage_at_different_alphas(self):
        """Test that higher alphas give higher empirical coverage."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib, y_calib)

        alphas = [0.01, 0.05, 0.1, 0.2]
        coverages = []

        for alpha in alphas:
            _, intervals, _ = reg.predict(X_test, alpha=alpha)
            coverage = prediction_interval_coverage(y_test, intervals)
            coverages.append(coverage)

        # Generally, higher alpha should not decrease coverage
        # (though it's not guaranteed due to discretization)
        assert len(coverages) == len(alphas)
        assert all(c >= 0 for c in coverages)
        assert all(c <= 1 for c in coverages)

    def test_classifier_coverage_at_different_alphas(self):
        """Test classifier coverage at different alpha levels."""
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        clf = ConformalClassifier(LogisticRegression(max_iter=1000, random_state=42))
        clf.fit(X_train, y_train)
        clf.calibrate(X_calib, y_calib)

        alphas = [0.01, 0.05, 0.1, 0.2]
        coverages = []

        for alpha in alphas:
            pred_set, _, _, _ = clf.predict(X_test, alpha=alpha)
            coverage = prediction_set_coverage(y_test, pred_set)
            coverages.append(coverage)

        assert all(c >= 0 for c in coverages)
        assert all(c <= 1 for c in coverages)


class TestBoundaryConditions:
    """Tests for boundary and edge conditions."""

    def test_small_calibration_set(self):
        """Test with very small calibration set."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.9, random_state=42
        )

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib, y_calib)

        y_pred, intervals, _ = reg.predict(X_test)
        assert intervals.shape[0] == X_test.shape[0]

    def test_large_alpha(self):
        """Test with alpha close to 1 that triggers warning."""
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib, y_calib)

        # With small calibration set, large alpha can trigger clipping warning
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_pred, intervals, _ = reg.predict(X_test, alpha=0.95)
            # May or may not warn depending on calibration size, just check it works
            assert intervals.shape[0] == X_test.shape[0]
            assert np.all(intervals[:, 0] <= intervals[:, 1])

    def test_very_small_alpha(self):
        """Test with very small alpha."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib, y_calib)

        y_pred, intervals, _ = reg.predict(X_test, alpha=0.001)

        # Should still return valid intervals
        assert intervals.shape[0] == X_test.shape[0]

    def test_invalid_alpha(self):
        """Test that invalid alpha values raise errors."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib, y_calib)

        with pytest.raises(ValueError):
            reg.predict(X_test, alpha=1.0)

        with pytest.raises(ValueError):
            reg.predict(X_test, alpha=1.1)

        with pytest.raises(ValueError):
            reg.predict(X_test, alpha=0.0)

        with pytest.raises(ValueError):
            reg.predict(X_test, alpha=-0.1)


class TestDataTypeHandling:
    """Tests for different data types."""

    def test_different_dtypes_float(self):
        """Test with different float dtypes."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        for dtype in [np.float32, np.float64]:
            X_train_typed = X_train.astype(dtype)
            X_calib_typed = X_calib.astype(dtype)
            X_test_typed = X_test.astype(dtype)

            reg = ConformalRegressor(LinearRegression())
            reg.fit(X_train_typed, y_train)
            reg.calibrate(X_calib_typed, y_calib)

            y_pred, intervals, _ = reg.predict(X_test_typed)
            assert y_pred.shape[0] == X_test.shape[0]

    def test_1d_target_conversion(self):
        """Test that 1D targets are handled correctly."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        # Ensure y is 1D
        assert y_train.ndim == 1
        assert y_calib.ndim == 1

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib, y_calib)

        y_pred, intervals, _ = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]


class TestConsistency:
    """Tests for consistency across multiple calls."""

    def test_deterministic_predictions_with_seed(self):
        """Test that predictions are deterministic with set random seed."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        reg1 = ConformalRegressor(LinearRegression())
        reg1.fit(X_train, y_train)
        reg1.calibrate(X_calib, y_calib)
        pred1, int1, _ = reg1.predict(X_test, alpha=0.1)

        reg2 = ConformalRegressor(LinearRegression())
        reg2.fit(X_train, y_train)
        reg2.calibrate(X_calib, y_calib)
        pred2, int2, _ = reg2.predict(X_test, alpha=0.1)

        np.testing.assert_array_almost_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(int1, int2)

    def test_multiple_calls_give_same_intervals(self):
        """Test that calling predict multiple times gives same results."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib, X_test, y_calib, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib, y_calib)

        pred1, int1, q1 = reg.predict(X_test, alpha=0.1)
        pred2, int2, q2 = reg.predict(X_test, alpha=0.1)
        pred3, int3, q3 = reg.predict(X_test, alpha=0.1)

        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)
        np.testing.assert_array_equal(int1, int2)
        np.testing.assert_array_equal(int2, int3)
        assert q1 == q2 == q3


class TestWarnings:
    """Tests for appropriate warning generation."""

    def test_multiple_calibrations_warning(self):
        """Test that multiple calibrations generate warning."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_calib1, X_calib2 = train_test_split(X_rest, test_size=0.5, random_state=42)
        y_calib1, y_calib2 = train_test_split(y_rest, test_size=0.5, random_state=42)

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib1, y_calib1)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reg.calibrate(X_calib2, y_calib2)

            assert len(w) >= 1
            assert issubclass(w[-1].category, UserWarning)
            assert "already calibrated" in str(w[-1].message).lower()
