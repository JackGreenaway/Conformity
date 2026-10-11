# Cross-validation

[Download the complete Python example](downloads/cross-validation.py). Run it with `python cross-validation.py` in the development environment.

## Question

Can ordinary scikit-learn model selection coexist with uncertainty calibration? Constructor options survive cloning, and `score` uses point performance. Every fold may calibrate its own clone, but final calibration must be independent of the procedure that selected the model.

This example wraps the whole pipeline because `cross_validate` calls ordinary `fit` on each clone, and we want that call to hold out calibration before the scaler learns anything. A wrapper on the final Ridge model would split only after the outer scaler had fitted on all fold-training rows, including the internal calibration features. That layout still supports point scoring, but the usual split-conformal proof would not apply to its uncertainty output. For a final-step wrapper with learned preprocessing, manage the calibration split before fitting each fold, or use that layout for your final model with a separate held-out calibration set. See [placement choices](../user-guide/estimators.md#why-the-split-matters).

```python
import numpy as np
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
    Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]), random_state=52
)
results = cross_validate(
    model,
    X,
    y,
    cv=KFold(3, shuffle=True, random_state=53),
    return_estimator=True,
    error_score="raise",
)
assert not hasattr(model, "estimator_")
assert len(results["estimator"]) == 3
assert all(fold.n_calibration_ == 40 for fold in results["estimator"])
assert np.isfinite(results["test_score"]).all()
print("Fold point R²:", np.round(results["test_score"], 3))
```

<!-- BEGIN EXAMPLE OUTPUT -->

```text
Fold point R²: [0.566 0.428 0.595]
```

<!-- END EXAMPLE OUTPUT -->

## Configuration and interpretation

The fold R² values are approximately 0.566, 0.428 and 0.595. This variation describes point-model performance, not conformal coverage.

Each fold fits on 200 rows, internally reserves 40 for calibration, and reports point R² on its validation fold. Learned preprocessing belongs inside the wrapper so its statistics exclude those 40 rows. Cloning leaves the original model unfitted.

## Next steps

Choose point-model settings using development data. Fit the selected model and calibrate on fresh independent observations; test uncertainty quality once on an untouched partition. Fold point scores alone do not validate final coverage. See [estimator guidance](../user-guide/estimators.md).
