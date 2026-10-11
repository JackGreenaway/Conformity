"""Tests for boolean prediction sets and interval diagnostics."""

from typing import Any

from numpy.typing import ArrayLike
import numpy as np
import pytest

from conformity.metrics import (
    interval_score,
    prediction_interval_coverage,
    prediction_interval_width,
    prediction_set_coverage,
    prediction_set_size,
    prediction_set_empty_rate,
    prediction_set_singleton_rate,
)


@pytest.mark.parametrize(
    "classes", [[0, 1, 2], ["cat", "dog", "fox"], [2**60 + 1, 2**60 + 2, 2**60 + 3]]
)
def test_boolean_set_metrics(classes: list[Any]) -> None:
    """Measure membership without encoding labels in the set matrix."""
    mask = np.array([[True, False, False], [False, False, False], [True, True, True]])
    assert prediction_set_coverage(classes, mask, classes=classes) == pytest.approx(
        2 / 3
    )
    assert prediction_set_size(mask) == pytest.approx(4 / 3)
    assert prediction_set_empty_rate(mask) == pytest.approx(1 / 3)
    assert prediction_set_singleton_rate(mask) == pytest.approx(1 / 3)


@pytest.mark.parametrize("values", [[[0, 1]], [[0, np.nan]], [["cat", None]]])
def test_label_encoded_sets_are_rejected(values: ArrayLike) -> None:
    """Only boolean membership matrices are supported."""
    with pytest.raises(ValueError, match="boolean"):
        prediction_set_size(values)


def test_weighted_interval_diagnostics() -> None:
    """Weights affect width and score consistently."""
    bounds = [[0, 2], [1, 5]]
    assert prediction_interval_width(bounds, sample_weight=[3, 1]) == 2.5
    assert interval_score([1, 6], bounds, alpha=0.2, sample_weight=[3, 1]) == 5


class TestPredictionIntervalCoverage:
    """Tests for prediction_interval_coverage metric."""

    def test_perfect_coverage(self) -> None:
        """Test when all true values are within intervals."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        intervals = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 1.0

    def test_no_coverage(self) -> None:
        """Test when no true values are within intervals."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        intervals = np.array([[10.0, 11.0], [11.0, 12.0], [12.0, 13.0], [13.0, 14.0]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 0.0

    def test_partial_coverage(self) -> None:
        """Test with partial coverage."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        intervals = np.array([[0.5, 1.5], [5.0, 6.0], [2.5, 3.5], [3.5, 4.5]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 0.75

    def test_boundary_conditions(self) -> None:
        """Test with values on interval boundaries."""
        y_true = np.array([1.0, 2.0, 3.0])
        intervals = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        coverage = prediction_interval_coverage(y_true, intervals)
        assert coverage == 1.0
