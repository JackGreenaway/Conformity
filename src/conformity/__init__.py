"""
Conformity: Conformal prediction for regression and classification.

This package provides tools for conformal prediction, enabling reliable
uncertainty quantification in machine learning models with statistical guarantees.
"""

from .base import BaseConformalPredictor
from .classifier import ConformalClassifier
from .regressor import ConformalRegressor
from .metrics import (
    prediction_set_coverage,
    prediction_set_efficiency,
    prediction_interval_coverage,
    prediction_interval_efficiency,
    prediction_interval_ratio,
    prediction_interval_mse,
)

__version__ = "0.1.0"

__all__ = [
    "BaseConformalPredictor",
    "ConformalClassifier",
    "ConformalRegressor",
    "prediction_set_coverage",
    "prediction_set_efficiency",
    "prediction_interval_coverage",
    "prediction_interval_efficiency",
    "prediction_interval_ratio",
    "prediction_interval_mse",
]
