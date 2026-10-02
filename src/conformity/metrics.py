"""Validated diagnostics for conformal sets and intervals.

Weights affect evaluation only, not the calibration coverage guarantee.
Boolean sets use columns in an explicitly supplied ``classes`` order for coverage.
Legacy label arrays use NaN (or None for object arrays) for exclusions.
"""

import numpy as np
from .base import validate_alpha


def _average(values, sample_weight=None):
    values = np.asarray(values, dtype=float)
    if sample_weight is None:
        return float(np.mean(values))
    weight = np.asarray(sample_weight, dtype=float)
    if (
        weight.shape != values.shape
        or not np.isfinite(weight).all()
        or (weight < 0).any()
        or weight.sum() <= 0
    ):
        raise ValueError("sample_weight must be finite, nonnegative and match samples")
    # Zero-weight infinite widths must not contaminate the result with 0 * inf.
    positive = weight > 0
    return float(np.average(values[positive], weights=weight[positive]))


def _vector(values, n, name):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"{name} must be a vector matching the number of samples")
    return values


def _intervals(intervals):
    intervals = np.asarray(intervals, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals):
        raise ValueError("prediction_intervals must have nonempty shape (n_samples, 2)")
    if np.isnan(intervals).any() or (intervals[:, 0] > intervals[:, 1]).any():
        raise ValueError("interval bounds must be ordered and contain no NaN")
    if np.isposinf(intervals[:, 0]).any() or np.isneginf(intervals[:, 1]).any():
        raise ValueError("infinite bounds must point outward")
    return intervals


def _targets(y, n):
    y = np.asarray(_vector(y, n, "y_true"), dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("y_true must be finite")
    return y


def _sets(prediction_set):
    values = np.asarray(prediction_set)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("prediction_set must be a nonempty 2D array")
    if values.dtype == bool:
        return values, values
    if np.issubdtype(values.dtype, np.number):
        mask = ~np.isnan(values)
    else:
        mask = np.fromiter(
            (
                v is not None
                and not (isinstance(v, (float, np.floating)) and np.isnan(v))
                for v in values.flat
            ),
            dtype=bool,
            count=values.size,
        ).reshape(values.shape)
    return values, mask


def prediction_set_coverage(
    y_true, prediction_set, *, classes=None, sample_weight=None
):
    """Fraction of sets containing the true class; supports arbitrary labels."""
    values, mask = _sets(prediction_set)
    y = _vector(y_true, len(values), "y_true")
    if values.dtype == bool:
        if classes is None:
            raise ValueError("classes is required for boolean prediction sets")
        classes = _vector(classes, values.shape[1], "classes")
        if len(np.unique(classes)) != len(classes):
            raise ValueError("classes must be unique")
        hits = mask & (y[:, None] == classes[None, :])
    else:
        hits = mask & (values == y[:, None])
    return _average(hits.any(axis=1), sample_weight)


def prediction_set_size(prediction_set, *, sample_weight=None):
    """Mean cardinality (smaller is more efficient at comparable coverage)."""
    return _average(_sets(prediction_set)[1].sum(axis=1), sample_weight)


def prediction_set_empty_rate(prediction_set, *, sample_weight=None):
    """Fraction of empty sets."""
    return _average(_sets(prediction_set)[1].sum(axis=1) == 0, sample_weight)


def prediction_set_singleton_rate(prediction_set, *, sample_weight=None):
    """Fraction of sets containing exactly one label."""
    return _average(_sets(prediction_set)[1].sum(axis=1) == 1, sample_weight)


def prediction_set_efficiency(prediction_set, *, sample_weight=None):
    """Legacy (size - 1) / (number of classes - 1) diagnostic.

    Empty sets can yield negative values. For a single class, return mean size
    rather than divide by zero. Prefer prediction_set_size for interpretation.
    """
    values, mask = _sets(prediction_set)
    size = mask.sum(axis=1)
    return _average(
        size if values.shape[1] == 1 else (size - 1) / (values.shape[1] - 1),
        sample_weight,
    )


def prediction_interval_coverage(y_true, prediction_intervals, *, sample_weight=None):
    """Fraction of targets contained in closed prediction intervals."""
    intervals = _intervals(prediction_intervals)
    y = _targets(y_true, len(intervals))
    return _average((intervals[:, 0] <= y) & (y <= intervals[:, 1]), sample_weight)


def prediction_interval_width(prediction_intervals, *, sample_weight=None):
    """Mean interval width; unbounded intervals have infinite width."""
    intervals = _intervals(prediction_intervals)
    return _average(intervals[:, 1] - intervals[:, 0], sample_weight)


def prediction_interval_efficiency(
    point_prediction, prediction_intervals, relative=False, *, sample_weight=None
):
    """Mean width, optionally divided by abs(point_prediction) + 1e-10."""
    intervals = _intervals(prediction_intervals)
    point = _targets(point_prediction, len(intervals))
    width = intervals[:, 1] - intervals[:, 0]
    return _average(
        width / (np.abs(point) + 1e-10) if relative else width, sample_weight
    )


def prediction_interval_ratio(
    point_predictions, prediction_intervals, *, sample_weight=None
):
    """Mean upper bound / point prediction; undefined for zero predictions."""
    intervals = _intervals(prediction_intervals)
    point = _targets(point_predictions, len(intervals))
    if (point == 0).any():
        raise ValueError("interval ratio is undefined for zero point predictions")
    return _average(intervals[:, 1] / point, sample_weight)


def prediction_interval_mse(y_true, prediction_intervals, *, sample_weight=None):
    """MSE of each bound; infinite bounds give infinite MSE."""
    intervals = _intervals(prediction_intervals)
    y = _targets(y_true, len(intervals))
    return tuple(_average((intervals[:, i] - y) ** 2, sample_weight) for i in range(2))


def interval_score(y_true, prediction_intervals, alpha=0.05, *, sample_weight=None):
    """Mean Winkler score: width plus 2/alpha times each missed-bound distance.

    Lower is better; combines sharpness and miscoverage. Infinite intervals
    have infinite score. Alpha must match that used to construct intervals.
    """
    alpha = validate_alpha(alpha)
    intervals = _intervals(prediction_intervals)
    y = _targets(y_true, len(intervals))
    score = (
        intervals[:, 1]
        - intervals[:, 0]
        + 2 / alpha * np.maximum(intervals[:, 0] - y, 0)
        + 2 / alpha * np.maximum(y - intervals[:, 1], 0)
    )
    return _average(score, sample_weight)
