"""
Conformity: Conformal prediction for regression and classification.

This package provides tools for conformal prediction, enabling reliable
uncertainty quantification in machine learning models with statistical guarantees.
"""

from __future__ import annotations

from .base import BaseConformalPredictor
from .classifier import ConformalClassifier
from .metrics import (
    interval_score,
    normalized_prediction_interval_width,
    prediction_interval_coverage,
    prediction_interval_lower_miss_rate,
    prediction_interval_upper_miss_rate,
    prediction_interval_width,
    prediction_set_coverage,
    prediction_set_coverage_by_class,
    prediction_set_empty_rate,
    prediction_set_singleton_rate,
    prediction_set_size,
)
from .regressor import ConformalRegressor

__version__: str = "0.1.11"

__all__: list[str] = [
    "BaseConformalPredictor",
    "ConformalClassifier",
    "ConformalRegressor",
    "prediction_set_coverage",
    "prediction_set_coverage_by_class",
    "prediction_interval_coverage",
    "prediction_interval_lower_miss_rate",
    "prediction_interval_upper_miss_rate",
    "prediction_interval_width",
    "normalized_prediction_interval_width",
    "interval_score",
    "prediction_set_size",
    "prediction_set_empty_rate",
    "prediction_set_singleton_rate",
]
