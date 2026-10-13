"""Run cross-validation with conformal regression metrics."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalRegressor

rng = np.random.default_rng(51)
X = rng.normal(size=(300, 3))
y = X[:, 0] + rng.normal(size=300)
# Wrap the whole pipeline here so each automatic fold split precedes scaling.
model = ConformalRegressor(
    Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
    tts_kwargs={"random_state": 52},
)
alpha = 0.1


def evaluate_metric(
    name: str,
) -> Callable[[BaseEstimator, ArrayLike, ArrayLike], float]:
    """Create a scorer that selects a metric from the fold evaluation."""

    def scorer(
        estimator: BaseEstimator, X_fold_test: ArrayLike, y_fold_test: ArrayLike
    ) -> float:
        return float(estimator.evaluate(X_fold_test, y_fold_test, alpha=alpha)[name])

    return scorer


results = cross_validate(
    model,
    X,
    y,
    cv=KFold(3, shuffle=True, random_state=53),
    scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "coverage": evaluate_metric("coverage"),
        "lower_miss_rate": evaluate_metric("lower_miss_rate"),
        "upper_miss_rate": evaluate_metric("upper_miss_rate"),
        "mean_width": evaluate_metric("mean_width"),
        "normalized_mean_width": evaluate_metric("normalized_mean_width"),
        "interval_score": evaluate_metric("interval_score"),
    },
    return_estimator=True,
    error_score="raise",
)
assert not hasattr(model, "estimator_")
assert len(results["estimator"]) == 3
assert all(fold.n_calibration_ == 40 for fold in results["estimator"])
assert np.isfinite(results["test_r2"]).all()
print("Fold point R²:", np.round(results["test_r2"], 3))
print("Fold point MAE:", np.round(-results["test_mae"], 3))
print("Fold coverage:", np.round(results["test_coverage"], 3))
print("Fold lower-tail misses:", np.round(results["test_lower_miss_rate"], 3))
print("Fold upper-tail misses:", np.round(results["test_upper_miss_rate"], 3))
print("Fold mean width:", np.round(results["test_mean_width"], 3))
print(
    "Fold normalized mean width:",
    np.round(results["test_normalized_mean_width"], 3),
)
print("Fold interval score:", np.round(results["test_interval_score"], 3))
