# Getting started

## Question: how uncertain is a numeric prediction?

A regression score measures average model quality, but does not provide a range for the next outcome. Use held-out residuals to produce a prediction interval. This example uses synthetic data so you can run it without downloads.

## Configuration

The test split is reserved for evaluation. The conformal wrapper encloses the full pipeline and automatically holds out the default 20% calibration partition before the scaler or Ridge fit. Set `calibration_splitter=ShuffleSplit(n_splits=1, test_size=0.25)` to reserve a different fraction. `alpha=0.1` requests 90% marginal coverage; it is not the Ridge regularization parameter.

**Notice:** When using automatic calibration with learned preprocessing, wrap the entire pipeline. Putting the wrapper after a scaler lets the scaler learn from the internal calibration features, which invalidates the usual split-conformal coverage argument. The [pipeline guide](user-guide/estimators.md#why-the-split-matters) explains manual calibration and other placement choices.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalRegressor, prediction_interval_coverage

X, y = make_regression(n_samples=1000, n_features=5, noise=15, random_state=7)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
model = ConformalRegressor(
    Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
    calibration_splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=9),
).fit(X_dev, y_dev)
alpha = 0.1
points, intervals = model.predict(X_test, alpha=alpha, return_interval=True)
coverage = prediction_interval_coverage(y_test, intervals)
assert points.shape == (250,)
assert intervals.shape == (250, 2)
assert np.all(intervals[:, 0] <= points)
assert np.all(points <= intervals[:, 1])
assert 0 <= coverage <= 1
print(
    {
        "coverage": round(coverage, 3),
        "width": round(
            model.evaluate(X_test, y_test, alpha)["mean_width"],
            3,
        ),
    }
)
```

## Results and interpretation

The result has 250 point predictions and a `(250, 2)` lower/upper bound array. The printed coverage is the fraction of test targets inside their intervals, while width is in target units. Coverage varies with the split and model; 90% is a population guarantee under the assumptions, not a requirement that every finite test sample achieves exactly 90%.

The default intervals are symmetric and have the same width for every row at a fixed alpha. `ConformalRegressor(interval_method="asymmetric")` instead calibrates lower and upper residual corrections separately while retaining the same marginal coverage target. Neither option adapts interval width to each row. Better point predictions can reduce calibration residuals and make intervals more useful. Smaller alpha generally enlarges regions; it cannot fix distribution shift.

## Next steps

[Plot intervals](plotting.md), compare [diagnostics](user-guide/diagnostics.md), or use [classification sets](examples/classification.md) for categorical targets. Read [pipeline placement](user-guide/estimators.md) before tuning or adding preprocessing.
