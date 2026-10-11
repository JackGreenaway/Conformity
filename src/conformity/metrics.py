"""Validated diagnostics for conformal sets and intervals.

Weights affect evaluation only, not the calibration coverage guarantee.
Boolean sets use columns in an explicitly supplied ``classes`` order for coverage.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._typing import FloatArray
from .base import validate_alpha


def _average(values: ArrayLike, sample_weight: Optional[ArrayLike] = None) -> float:
    """Compute a validated weighted mean, ignoring zero-weight infinite values."""
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


def _vector(values: ArrayLike, n: int, name: str) -> NDArray[Any]:
    """Require a one-dimensional vector of the specified sample count."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"{name} must be a vector matching the number of samples")
    return values


def _intervals(intervals: ArrayLike) -> FloatArray:
    """Require nonempty ordered bounds, permitting outward infinities."""
    intervals = np.asarray(intervals, dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals):
        raise ValueError("prediction_intervals must have nonempty shape (n_samples, 2)")
    if np.isnan(intervals).any() or (intervals[:, 0] > intervals[:, 1]).any():
        raise ValueError("interval bounds must be ordered and contain no NaN")
    if np.isposinf(intervals[:, 0]).any() or np.isneginf(intervals[:, 1]).any():
        raise ValueError("infinite bounds must point outward")
    return intervals


def _targets(y: ArrayLike, n: int) -> FloatArray:
    """Require finite numeric targets matching the interval sample count."""
    y = np.asarray(_vector(y, n, "y_true"), dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("y_true must be finite")
    return y


def _sets(prediction_set: ArrayLike) -> NDArray[np.bool_]:
    """Require a nonempty boolean membership matrix."""
    values = np.asarray(prediction_set)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("prediction_set must be a nonempty 2D array")
    if values.dtype != bool:
        raise ValueError("prediction_set must be a boolean membership matrix")
    return values


def prediction_set_coverage(
    y_true: ArrayLike,
    prediction_set: ArrayLike,
    *,
    classes: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Fraction of sets containing the true class; supports arbitrary labels.

    Parameters
    ----------
    y_true
        One observed target or class label per sample.
    prediction_set
        Boolean membership matrix in classes order.
    classes
        Unique class labels in boolean membership-column order.
    sample_weight
        Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
    Returns
    -------
    result : float
        Fraction of samples whose set contains the true label; weighted when requested.

    Examples
    --------
    >>> from conformity import prediction_set_coverage
    >>> sets = [[True, False], [False, True], [True, True]]
    >>> prediction_set_coverage(["billing", "account", "billing"], sets,
    ...                         classes=["billing", "account"])
    1.0
    """
    mask = _sets(prediction_set)
    y = _vector(y_true, len(mask), "y_true")
    if classes is None:
        raise ValueError("classes is required for boolean prediction sets")
    classes = _vector(classes, mask.shape[1], "classes")
    if len(np.unique(classes)) != len(classes):
        raise ValueError("classes must be unique")
    hits = mask & (y[:, None] == classes[None, :])
    return _average(hits.any(axis=1), sample_weight)


def prediction_set_size(
    prediction_set: ArrayLike, *, sample_weight: Optional[ArrayLike] = None
) -> float:
    """Mean cardinality (smaller is more efficient at comparable coverage).

    Parameters
    ----------
    prediction_set
        Boolean membership matrix in classes order.
    sample_weight
        Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
    Returns
    -------
    result : float
        Mean number of included labels per sample; weighted when requested.

    Examples
    --------
    >>> from conformity import prediction_set_size
    >>> prediction_set_size([[True, False], [False, False], [True, True]])
    1.0
    """
    return _average(_sets(prediction_set).sum(axis=1), sample_weight)


def prediction_set_empty_rate(
    prediction_set: ArrayLike, *, sample_weight: Optional[ArrayLike] = None
) -> float:
    """Fraction of empty sets.

    Parameters
    ----------
    prediction_set
        Boolean membership matrix in classes order.
    sample_weight
        Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
    Returns
    -------
    result : float
        Fraction of samples with no included labels; weighted when requested.

    Examples
    --------
    >>> from conformity import prediction_set_empty_rate
    >>> prediction_set_empty_rate([[True, False], [False, False]])
    0.5
    """
    return _average(_sets(prediction_set).sum(axis=1) == 0, sample_weight)


def prediction_set_singleton_rate(
    prediction_set: ArrayLike, *, sample_weight: Optional[ArrayLike] = None
) -> float:
    """Fraction of sets containing exactly one label.

    Parameters
    ----------
    prediction_set
        Boolean membership matrix in classes order.
    sample_weight
        Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
    Returns
    -------
    result : float
        Fraction of samples with exactly one included label; weighted when requested.

    Examples
    --------
    >>> from conformity import prediction_set_singleton_rate
    >>> prediction_set_singleton_rate([[True, False], [True, True]])
    0.5
    """
    return _average(_sets(prediction_set).sum(axis=1) == 1, sample_weight)


def prediction_interval_coverage(
    y_true: ArrayLike,
    prediction_intervals: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Fraction of targets contained in closed prediction intervals.

    Parameters
    ----------
    y_true
        One observed target or class label per sample.
    prediction_intervals
        Ordered lower/upper bounds of shape (n_samples, 2).
    sample_weight
        Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
    Returns
    -------
    result : float
        Fraction of targets inside their closed interval; weighted when requested.

    Examples
    --------
    >>> from conformity import prediction_interval_coverage
    >>> prediction_interval_coverage([1.0, 4.0], [[0.0, 2.0], [1.0, 3.0]])
    0.5
    """
    intervals = _intervals(prediction_intervals)
    y = _targets(y_true, len(intervals))
    return _average((intervals[:, 0] <= y) & (y <= intervals[:, 1]), sample_weight)


def prediction_interval_width(
    prediction_intervals: ArrayLike, *, sample_weight: Optional[ArrayLike] = None
) -> float:
    """Mean interval width; unbounded intervals have infinite width.

    Parameters
    ----------
    prediction_intervals
        Ordered lower/upper bounds of shape (n_samples, 2).
    sample_weight
        Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
    Returns
    -------
    result : float
        Mean upper-minus-lower bound width; may be infinite.

    Examples
    --------
    >>> from conformity import prediction_interval_width
    >>> prediction_interval_width([[0.0, 2.0], [1.0, 3.0]])
    2.0
    """
    intervals = _intervals(prediction_intervals)
    return _average(intervals[:, 1] - intervals[:, 0], sample_weight)


def interval_score(
    y_true: ArrayLike,
    prediction_intervals: ArrayLike,
    alpha: float = 0.05,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Mean Winkler score: width plus 2/alpha times each missed-bound distance.

        Lower is better; combines sharpness and miscoverage. Infinite intervals
        have infinite score. Alpha must match that used to construct intervals.

    Parameters
    ----------
    y_true
        One observed target or class label per sample.
    prediction_intervals
        Ordered lower/upper bounds of shape (n_samples, 2).
    alpha
        Miscoverage probability strictly between zero and one.
    sample_weight
        Optional finite, nonnegative evaluation or fitting weights; calibration is unweighted.
    Returns
    -------
    result : float
        Mean interval score; may be infinite. Smaller values indicate a better score.

    Examples
    --------
    >>> from conformity import interval_score
    >>> interval_score([1.0, 4.0], [[0.0, 2.0], [1.0, 3.0]], alpha=0.1)
    12.0
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
