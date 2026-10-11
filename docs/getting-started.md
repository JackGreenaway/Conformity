# Getting started

## Question: how uncertain is a numeric prediction?

A regression score measures average model quality, but does not provide a range for the next outcome. Use held-out residuals to produce a prediction interval. This example uses synthetic data so you can run it without downloads.

## Configuration

The test split is reserved for evaluation. We hold out 25% of development rows for calibration before fitting the pipeline. The scaler learns only from training rows, and the final conformal step wraps Ridge. Calibrate that step using calibration features transformed by the fitted scaler. `alpha=0.1` requests 90% marginal coverage; it is not the Ridge regularization parameter.

Use this layout when you want preprocessing to remain ordinary pipeline steps and can reserve calibration data yourself. `auto_calibrate=False` keeps the wrapper from making another split. If you want `fit` to split automatically, wrap the whole pipeline so that calibration is excluded before the scaler fits. The [pipeline guide](user-guide/estimators.md#why-the-split-matters) explains both choices and when automatic calibration on the final step is safe.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalRegressor, prediction_interval_coverage

X, y = make_regression(n_samples=1000, n_features=5, noise=15, random_state=7)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=9
)
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("conformal", ConformalRegressor(Ridge(alpha=1.0), auto_calibrate=False)),
    ]
).fit(X_train, y_train)
conformal = model.named_steps["conformal"]
conformal.calibrate(model[:-1].transform(X_calib), y_calib)
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
            conformal.evaluate(model[:-1].transform(X_test), y_test, alpha)[
                "mean_width"
            ],
            3,
        ),
    }
)
```

## Results and interpretation

The result has 250 point predictions and a `(250, 2)` lower/upper bound array. The printed coverage is the fraction of test targets inside their intervals, while width is in target units. Coverage varies with the split and model; 90% is a population guarantee under the assumptions, not a requirement that every finite test sample achieves exactly 90%.

Intervals are symmetric and have the same width for every row at a fixed alpha. Better point predictions can reduce calibration residuals and make intervals more useful. Smaller alpha generally enlarges regions; it cannot fix distribution shift.

## Next steps

[Plot intervals](plotting.md), compare [diagnostics](user-guide/diagnostics.md), or use [classification sets](examples/classification.md) for categorical targets. Read [pipeline placement](user-guide/estimators.md) before tuning or adding preprocessing.
