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
    prediction_interval_coverage,
    prediction_interval_efficiency,
    prediction_interval_mse,
    prediction_interval_ratio,
    prediction_interval_width,
    prediction_set_coverage,
    prediction_set_efficiency,
    prediction_set_empty_rate,
    prediction_set_singleton_rate,
    prediction_set_size,
)
from .regressor import ConformalRegressor

__version__: str = "0.1.1"

__all__: list[str] = [
    "BaseConformalPredictor",
    "ConformalClassifier",
    "ConformalRegressor",
    "prediction_set_coverage",
    "prediction_set_efficiency",
    "prediction_interval_coverage",
    "prediction_interval_efficiency",
    "prediction_interval_ratio",
    "prediction_interval_mse",
    "prediction_interval_width",
    "interval_score",
    "prediction_set_size",
    "prediction_set_empty_rate",
    "prediction_set_singleton_rate",
]
