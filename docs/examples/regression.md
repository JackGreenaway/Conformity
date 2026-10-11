# Regression decisions

## Question

How can a decision use a predicted range rather than a single estimate? Start with the runnable [getting started workflow](../getting-started.md). Here a simpler synthetic relation makes the calibration choices visible.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

from conformity import ConformalRegressor

rng = np.random.default_rng(12)
X = rng.normal(size=(600, 2))
y = 3 * X[:, 0] + rng.normal(size=600)
model = ConformalRegressor(LinearRegression(), auto_calibrate=False).fit(
    X[:300], y[:300]
)
model.calibrate(X[300:450], y[300:450])
intervals = model.predict_interval(X[450:], alpha=0.1)
report = model.evaluate(X[450:], y[450:], alpha=0.1)
needs_review = (intervals[:, 0] <= 0) & (intervals[:, 1] >= 0)
assert needs_review.shape == (150,)
assert model.n_calibration_ == 150
assert 0 <= report["coverage"] <= 1
print(report)
print("Intervals crossing the decision boundary:", int(needs_review.sum()))
```

## Configuration choices

Explicit fit/calibration/test slices keep all three roles disjoint. `auto_calibrate=False` avoids making a second internal split. A boundary of zero is an illustrative policy, not a package default.

## Results and interpretation

The report pairs point error with coverage, width and interval score. An interval crossing zero suggests the outcome could plausibly lie on either side. This rule does not give conditional coverage for the reviewed subset or guarantee a business loss rate.

## Next steps

Choose a policy on development data and evaluate it once on untouched test data. Investigate group coverage and drift before deployment; see [diagnostics](../user-guide/diagnostics.md).

The [statistical specification](../legacy/THEORY.md#evaluation-and-limits) explains the coverage assumptions and limits behind this example.
