"""Run cross-validation with conformal regression metrics."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split,
    ShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalRegressor

rng = np.random.default_rng(51)
X = rng.normal(size=(300, 3))
y = X[:, 0] + rng.normal(size=300)
X, X_final, y, y_final = train_test_split(X, y, test_size=0.2, random_state=54)
X_calib, X_test, y_calib, y_test = train_test_split(
    X_final, y_final, test_size=0.5, random_state=55
)
# Wrap the whole pipeline here so each automatic fold split precedes scaling.
model = ConformalRegressor(
    Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
    calibration_splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=52),
)
alpha = 0.1
fold_splits = list(KFold(3, shuffle=True, random_state=53).split(X, y))


def evaluate_metric(
    name: str,
) -> Callable[[BaseEstimator, ArrayLike, ArrayLike], float]:
    """Create a scorer that selects a metric from the fold evaluation."""

    def scorer(
        estimator: BaseEstimator, X_fold_test: ArrayLike, y_fold_test: ArrayLike
    ) -> float:
        return float(estimator.evaluate(X_fold_test, y_fold_test, alpha=alpha)[name])

    return scorer


def custom_interval_cost(
    estimator: BaseEstimator, X_eval: ArrayLike, y_eval: ArrayLike
) -> float:
    """Penalize interval width plus a fixed cost for each missed target."""
    bounds = estimator.predict_interval(X_eval, alpha=alpha)
    targets = np.asarray(y_eval)
    missed = (targets < bounds[:, 0]) | (targets > bounds[:, 1])
    return float(np.mean(bounds[:, 1] - bounds[:, 0] + 10.0 * missed))


def negative_custom_interval_cost(
    estimator: BaseEstimator, X_eval: ArrayLike, y_eval: ArrayLike
) -> float:
    """Return negative cost because sklearn scorers are maximized."""
    return -custom_interval_cost(estimator, X_eval, y_eval)


results = cross_validate(
    model,
    X,
    y,
    cv=fold_splits,
    scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "coverage": evaluate_metric("coverage"),
        "lower_miss_rate": evaluate_metric("lower_miss_rate"),
        "upper_miss_rate": evaluate_metric("upper_miss_rate"),
        "mean_width": evaluate_metric("mean_width"),
        "normalized_mean_width": evaluate_metric("normalized_mean_width"),
        "interval_score": evaluate_metric("interval_score"),
        "custom_interval_cost": negative_custom_interval_cost,
    },
    return_estimator=True,
    error_score="raise",
)
assert not hasattr(model, "estimator_")
assert len(results["estimator"]) == 3
assert all(fold.n_calibration_ == 32 for fold in results["estimator"])
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
print(
    "Fold custom interval cost:",
    np.round(-results["test_custom_interval_cost"], 3),
)

# Manual calibration: cross_validate calls fit only, so split each fold's
# training rows explicitly, fit with auto_calibrate=False, then calibrate.
manual_reports = []
manual_n_calibration = []
for fold, (train_indices, test_indices) in enumerate(fold_splits):
    fit_positions, calibration_positions = train_test_split(
        np.arange(len(train_indices)), test_size=0.2, random_state=52 + fold
    )
    fit_indices = train_indices[fit_positions]
    calibration_indices = train_indices[calibration_positions]
    manual_model = ConformalRegressor(
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        auto_calibrate=False,
    ).fit(X[fit_indices], y[fit_indices])
    manual_model.calibrate(X[calibration_indices], y[calibration_indices])
    manual_reports.append(
        manual_model.evaluate(X[test_indices], y[test_indices], alpha=alpha)
    )
    manual_n_calibration.append(manual_model.n_calibration_)

print("Manual calibration rows per fold:", manual_n_calibration)
print(
    "Manual fold coverage:",
    np.round([report["coverage"] for report in manual_reports], 3),
)
print(
    "Manual fold mean width:",
    np.round([report["mean_width"] for report in manual_reports], 3),
)

# Final evaluation: cross-validation is for development. Fit the selected
# workflow on development rows, calibrate on fresh rows, and score once on test.
final_model = ConformalRegressor(
    Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
    auto_calibrate=False,
).fit(X, y)
final_model.calibrate(X_calib, y_calib)
final_report = final_model.evaluate(X_test, y_test, alpha=alpha)
print("Final calibration rows:", final_model.n_calibration_)
print("Final test R²:", round(final_report["r2"], 3))
print("Final test MAE:", round(final_report["mae"], 3))
print("Final test coverage:", round(final_report["coverage"], 3))
print("Final test mean width:", round(final_report["mean_width"], 3))
print(
    "Final test custom interval cost:",
    round(custom_interval_cost(final_model, X_test, y_test), 3),
)
