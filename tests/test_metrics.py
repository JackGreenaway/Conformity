"""Comprehensive tests for the metrics module."""

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification

from conformity.metrics import (
    prediction_set_coverage,
    prediction_set_efficiency,
    prediction_interval_coverage,
    prediction_interval_efficiency,
    prediction_interval_ratio,
    prediction_interval_mse,
)


class TestPredictionSetCoverage:
    """Tests for prediction_set_coverage metric."""

    def test_perfect_coverage(self):
        """Test when all samples are covered."""
        y_true = np.array([0, 1, 2])
        prediction_set = np.array([[0, 1], [1, 2], [2, 3]])
        coverage = prediction_set_coverage(y_true, prediction_set)
        assert coverage == 1.0

    def test_no_coverage(self):
        """Test when no samples are covered."""
        y_true = np.array([0, 1, 2])
        prediction_set = np.array([[3, 4], [5, 6], [7, 8]])
        coverage = prediction_set_coverage(y_true, prediction_set)
        assert coverage == 0.0

    def test_partial_coverage(self):
        """Test with partial coverage."""
        y_true = np.array([0, 1, 2, 3])
        prediction_set = np.array([[0, 1], [5, 6], [2, 3], [7, 8]])
        coverage = prediction_set_coverage(y_true, prediction_set)
        assert coverage == 0.5

    def test_with_nan_values(self):
        """Test handling of NaN values in prediction sets."""
        y_true = np.array([0, 1, 2])
        prediction_set = np.array([[0, 1], [np.nan, np.nan], [2, 3]])
        coverage = prediction_set_coverage(y_true, prediction_set)
        assert coverage == pytest.approx(2.0 / 3.0)

    def test_multiclass_predictions(self):
        """Test with multiclass predictions."""
        y_true = np.array([0, 1, 2, 1, 0])
        # Create properly structured array
        prediction_set = np.full((5, 3), np.nan)
        prediction_set[0] = [0, 1, 2]
        prediction_set[1, 0:2] = [1, 2]
        prediction_set[2] = [0, 1, 2]
        prediction_set[3] = [np.nan, np.nan, np.nan]
        prediction_set[4, 0:2] = [0, 1]
        coverage = prediction_set_coverage(y_true, prediction_set)
        assert coverage == 0.8


class TestPredictionSetEfficiency:
    """Tests for prediction_set_efficiency metric."""

    def test_all_singletons(self):
        """Test with single-element prediction sets (max efficiency)."""
        prediction_set = np.array([[0.0], [1.0], [2.0], [1.0]])
        # This case produces NaN because (1-1)/(1-1) = 0/0
        efficiency = prediction_set_efficiency(prediction_set)
        assert np.isnan(efficiency)  # Expected behavior for single-element sets

    def test_all_full_sets(self):
        """Test with full prediction sets (min efficiency)."""
        prediction_set = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        efficiency = prediction_set_efficiency(prediction_set)
        assert efficiency == 1.0

    def test_mixed_set_sizes(self):
        """Test with mixed set sizes."""
        # Create properly structured array with consistent shape
        prediction_set = np.full((3, 3), np.nan)
        prediction_set[0, 0] = 0
        prediction_set[1, 0:2] = [0, 1]
        prediction_set[2, 0:3] = [0, 1, 2]
        efficiency = prediction_set_efficiency(prediction_set)
        # Efficiency calculation: (count of non-nan - 1) / (total_cols - 1)
        # Row 0: (1-1)/(3-1) = 0; Row 1: (2-1)/(3-1) = 0.5; Row 2: (3-1)/(3-1) = 1.0
        expected = (0 + 0.5 + 1.0) / 3
        assert efficiency == pytest.approx(expected)

    def test_with_nan_values(self):
        """Test efficiency with NaN values representing empty sets."""
        prediction_set = np.array([
            [np.nan, np.nan, np.nan],
            [0.0, np.nan, np.nan],
            [0.0, 1.0, 2.0]
        ])
        efficiency = prediction_set_efficiency(prediction_set)
        # Row 0: (0-1)/(3-1) = -0.5; Row 1: (1-1)/(3-1) = 0; Row 2: (3-1)/(3-1) = 1.0
        # Mean = (-0.5 + 0 + 1.0) / 3 = 0.1667
        expected = (-0.5 + 0 + 1.0) / 3
        assert efficiency == pytest.approx(expected)


class TestPredictionIntervalCoverage:
    """Tests for prediction_interval_coverage metric."""

    def test_perfect_coverage(self):
        """Test when all true values are within intervals."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        intervals = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 1.0

    def test_no_coverage(self):
        """Test when no true values are within intervals."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        intervals = np.array([[10.0, 11.0], [11.0, 12.0], [12.0, 13.0], [13.0, 14.0]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 0.0

    def test_partial_coverage(self):
        """Test with partial coverage."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        intervals = np.array([[0.5, 1.5], [5.0, 6.0], [2.5, 3.5], [3.5, 4.5]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 0.75

    def test_boundary_conditions(self):
        """Test with values on interval boundaries."""
        y_true = np.array([1.0, 2.0, 3.0])
        intervals = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 1.0


class TestPredictionIntervalEfficiency:
    """Tests for prediction_interval_efficiency metric."""

    def test_equal_width_intervals(self):
        """Test with equal width intervals."""
        point_pred = np.array([1.0, 2.0, 3.0])
        intervals = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
        efficiency = prediction_interval_efficiency(point_pred, intervals)
        assert efficiency == 2.0

    def test_relative_efficiency(self):
        """Test relative efficiency normalization."""
        point_pred = np.array([1.0, 2.0, 10.0])
        intervals = np.array([[0.0, 2.0], [1.0, 3.0], [9.0, 11.0]])
        efficiency = prediction_interval_efficiency(point_pred, intervals, relative=True)
        expected = (2.0 / 1.0 + 2.0 / 2.0 + 2.0 / 10.0) / 3
        assert efficiency == pytest.approx(expected)

    def test_different_widths(self):
        """Test with varying interval widths."""
        point_pred = np.array([1.0, 2.0, 3.0])
        intervals = np.array([[0.5, 1.5], [1.0, 3.0], [2.9, 3.1]])
        efficiency = prediction_interval_efficiency(point_pred, intervals)
        expected = (1.0 + 2.0 + 0.2) / 3
        assert efficiency == pytest.approx(expected)


class TestPredictionIntervalRatio:
    """Tests for prediction_interval_ratio metric."""

    def test_symmetric_intervals(self):
        """Test with symmetric intervals around predictions."""
        point_pred = np.array([1.0, 2.0, 3.0])
        intervals = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
        ratio = prediction_interval_ratio(point_pred, intervals)
        # Ratio is upper_bound / point_pred: (2.0/1.0 + 3.0/2.0 + 4.0/3.0) / 3
        expected = (2.0 + 1.5 + 4.0/3.0) / 3
        assert ratio == pytest.approx(expected)

    def test_single_sample(self):
        """Test with single sample."""
        point_pred = np.array([5.0])
        intervals = np.array([[3.0, 7.0]])
        ratio = prediction_interval_ratio(point_pred, intervals)
        assert ratio == 1.4

    def test_varying_ratios(self):
        """Test with varying upper bound to prediction ratios."""
        point_pred = np.array([1.0, 2.0, 4.0])
        intervals = np.array([[0.0, 2.0], [0.0, 4.0], [0.0, 8.0]])
        ratio = prediction_interval_ratio(point_pred, intervals)
        expected = (2.0 + 2.0 + 2.0) / 3
        assert ratio == expected


class TestPredictionIntervalMSE:
    """Tests for prediction_interval_mse metric."""

    def test_perfect_predictions(self):
        """Test with perfect interval predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        intervals = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        mse_lower, mse_upper = prediction_interval_mse(y_true, intervals)
        assert mse_lower == 0.0
        assert mse_upper == 0.0

    def test_consistent_error(self):
        """Test with consistent prediction error."""
        y_true = np.array([1.0, 2.0, 3.0])
        intervals = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
        mse_lower, mse_upper = prediction_interval_mse(y_true, intervals)
        expected = 1.0  # (1 + 1 + 1) / 3
        assert mse_lower == expected
        assert mse_upper == expected

    def test_asymmetric_error(self):
        """Test with asymmetric lower and upper bound errors."""
        y_true = np.array([2.0, 3.0, 4.0])
        intervals = np.array([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
        mse_lower, mse_upper = prediction_interval_mse(y_true, intervals)
        expected_lower = (4.0 + 4.0 + 4.0) / 3
        expected_upper = (1.0 + 1.0 + 1.0) / 3
        assert mse_lower == expected_lower
        assert mse_upper == expected_upper


class TestMetricsWithRealData:
    """Integration tests for metrics with realistic data."""

    def test_metrics_with_regression_data(self):
        """Test metrics with actual regression data."""
        from sklearn.linear_model import LinearRegression
        from conformity.regressor import ConformalRegressor

        X, y = make_regression(n_samples=200, n_features=5, random_state=42)

        idx_train = np.random.RandomState(42).choice(200, 140, replace=False)
        idx_rest = np.setdiff1d(np.arange(200), idx_train)

        idx_calib = idx_rest[:30]
        idx_test = idx_rest[30:]

        X_train, y_train = X[idx_train], y[idx_train]
        X_calib, y_calib = X[idx_calib], y[idx_calib]
        X_test, y_test = X[idx_test], y[idx_test]

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train, y_train)
        reg.calibrate(X_calib, y_calib)

        y_pred, intervals, _ = reg.predict(X_test, alpha=0.1)

        coverage = prediction_interval_coverage(y_test, intervals)
        efficiency = prediction_interval_efficiency(y_pred, intervals)

        assert 0.80 <= coverage <= 1.0  # Coverage should be close to 90%
        assert efficiency > 0  # Should have non-zero width intervals

    def test_metrics_with_classification_data(self):
        """Test metrics with actual classification data."""
        from sklearn.linear_model import LogisticRegression
        from conformity.classifier import ConformalClassifier

        X, y = make_classification(
            n_samples=300, n_features=10, n_informative=8, random_state=42
        )

        idx_train = np.random.RandomState(42).choice(300, 210, replace=False)
        idx_rest = np.setdiff1d(np.arange(300), idx_train)

        idx_calib = idx_rest[:45]
        idx_test = idx_rest[45:]

        X_train, y_train = X[idx_train], y[idx_train]
        X_calib, y_calib = X[idx_calib], y[idx_calib]
        X_test, y_test = X[idx_test], y[idx_test]

        clf = ConformalClassifier(LogisticRegression(max_iter=1000, random_state=42))
        clf.fit(X_train, y_train)
        clf.calibrate(X_calib, y_calib)

        pred_set, _, _, _ = clf.predict(X_test, alpha=0.1)

        coverage = prediction_set_coverage(y_test, pred_set)
        efficiency = prediction_set_efficiency(pred_set)

        assert 0.80 <= coverage <= 1.0  # Coverage should be close to 90%
        assert 0.0 <= efficiency <= 1.0  # Efficiency in valid range


class TestMetricsEdgeCases:
    """Tests for edge cases in metrics."""

    def test_single_sample(self):
        """Test metrics with single sample."""
        y_true = np.array([1.0])
        intervals = np.array([[0.0, 2.0]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 1.0

    def test_large_dataset(self):
        """Test metrics with large dataset."""
        n_samples = 10000
        y_true = np.random.randn(n_samples)
        intervals = np.column_stack([y_true - 1, y_true + 1])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 1.0

    def test_negative_values(self):
        """Test with negative values."""
        y_true = np.array([-3.0, -2.0, -1.0])
        intervals = np.array([[-4.0, -2.0], [-3.0, -1.0], [-2.0, 0.0]])
        coverage = prediction_interval_coverage(y_true, intervals)
        # Check which are contained: -3 in [-4,-2]? Yes. -2 in [-3,-1]? Yes. -1 in [-2,0]? Yes.
        assert coverage == 1.0

    def test_zero_width_intervals(self):
        """Test with zero-width intervals."""
        point_pred = np.array([1.0, 2.0, 3.0])
        intervals = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        efficiency = prediction_interval_efficiency(point_pred, intervals)
        assert efficiency == 0.0
