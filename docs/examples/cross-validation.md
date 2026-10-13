# Cross-validation

[Download the complete Python example](downloads/cross-validation.py). Run it with `python cross-validation.py` in the development environment.

## Question

Can ordinary scikit-learn model selection coexist with uncertainty calibration? Constructor options survive cloning, and `score` uses point performance. Every fold may calibrate its own clone; custom scorers can also report coverage and efficiency on each held-out fold. Those cross-validation results are for model development, so final coverage still needs an untouched calibration set and test set.

This example wraps the whole pipeline because `cross_validate` calls ordinary `fit` on each clone, and we want that call to hold out calibration before the scaler learns anything. Custom scorers call `evaluate` on each fitted fold estimator and its validation rows. A wrapper on the final Ridge model would split only after the outer scaler had fitted on all fold-training rows, including the internal calibration features. See [placement choices](../user-guide/estimators.md#why-the-split-matters).

```python
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
        "coverage": evaluate_metric("coverage"),
        "mean_width": evaluate_metric("mean_width"),
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
print("Fold coverage:", np.round(results["test_coverage"], 3))
print("Fold mean width:", np.round(results["test_mean_width"], 3))
print("Fold interval score:", np.round(results["test_interval_score"], 3))
```

<!-- BEGIN EXAMPLE OUTPUT -->

```text
Fold point R²: [0.566 0.428 0.595]
Fold coverage: [0.89 0.76 0.85]
Fold mean width: [3.05  2.66  2.962]
Fold interval score: [3.571 5.323 3.989]
```

<!-- END EXAMPLE OUTPUT -->

## Configuration and interpretation

The fold R² values are approximately 0.566, 0.428 and 0.595. Fold coverage, mean width and interval score show the tradeoff between inclusion and efficiency on each held-out fold. With only 40 calibration rows per fold, coverage is coarse and can vary substantially around the 90% target.

Each fold fits on 200 rows, internally reserves 40 for calibration, and reports point and conformal metrics on its validation fold. The custom scorer receives the already fitted conformal estimator plus the validation features and targets, so `evaluate` can calculate coverage and efficiency directly. Learned preprocessing belongs inside the wrapper so its statistics exclude those 40 rows. Cloning leaves the original model unfitted.

## Next steps

Use fold metrics to compare candidate procedures during development, then select the procedure without consulting final calibration or test outcomes. Fit the selected model and calibrate on fresh independent observations; assess uncertainty quality once on an untouched test partition. Cross-validation metrics do not replace this final evaluation. See [estimator guidance](../user-guide/estimators.md).
