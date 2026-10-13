# Regression decisions

[Download the complete Python example](downloads/regression.py). Run it with `python regression.py` in the development environment.

## Question

Will a parcel arrive within 48 hours? A point estimate of 46 hours looks reassuring, but an interval spanning 40–52 hours exposes uncertainty around the deadline. This example simulates delivery times in hours from distance and parcel weight; it needs no downloaded dataset.

```python
"""Run the synthetic regression interval example."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from conformity import ConformalRegressor

rng = np.random.default_rng(12)
# Synthetic deliveries: distance in km and parcel weight in kg.
X = np.column_stack((rng.uniform(20, 500, 600), rng.uniform(0.5, 20, 600)))
y = 8 + 0.08 * X[:, 0] + 0.4 * X[:, 1] + rng.normal(0, 4, 600)

# Reserve 25% for final testing; split the remaining 75% into training/calibration.
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_dev, y_dev, test_size=1 / 3, random_state=14
)

# The explicit calibration split means the wrapper should not split again.
model = ConformalRegressor(LinearRegression(), auto_calibrate=False)
model.fit(X_train, y_train)
model.calibrate(X_calib, y_calib)
alpha = 0.1
points, intervals = model.predict(X_test, alpha=alpha, return_interval=True)
report = model.evaluate(X_test, y_test, alpha=alpha)

deadline_hours = 48.0
needs_review = (intervals[:, 0] <= deadline_hours) & (deadline_hours <= intervals[:, 1])
assert len(X_train) == 300
assert model.n_calibration_ == len(X_calib) == 150
assert needs_review.shape == (150,)
assert 0 <= report["coverage"] <= 1
for metric, value in report.items():
    print(f"{metric}: {value:.3f}")
print("Deliveries whose intervals cross 48 hours:", int(needs_review.sum()))
print("First three [observed, predicted, lower, upper] delivery times:")
print(np.round(np.column_stack((y_test, points, intervals))[:3], 1))
```

## Configuration choices

Two `train_test_split` calls create 300 training, 150 calibration and 150 test deliveries. The first reserves 25% for testing; the second reserves one third of the remaining 450 rows for calibration. Fixed seeds make the splits reproducible, and default shuffling chooses rows independently of their outcomes. `auto_calibrate=False` avoids making a second internal split. The 48-hour deadline is an illustrative delivery policy.

<!-- BEGIN EXAMPLE OUTPUT -->

```text
r2: 0.904
mae: 3.132
mse: 15.087
coverage: 0.953
lower_miss_rate: 0.027
upper_miss_rate: 0.020
mean_width: 14.531
normalized_mean_width: 0.704
interval_score: 15.937
Deliveries whose intervals cross 48 hours: 50
First three [observed, predicted, lower, upper] delivery times:
[[49.2 51.8 44.5 59. ]
 [20.1 27.  19.7 34.2]
 [53.9 45.3 38.1 52.6]]
```

<!-- END EXAMPLE OUTPUT -->

## Results and interpretation

This run covers 143 of 150 test outcomes (95.3%) with a mean width of 14.53 hours and flags 50 deliveries for review. Observed coverage can exceed or fall below the nominal 90% target on a finite test sample.

The report pairs point error with coverage, width and interval score, with widths expressed in hours. The printed rows compare actual delivery times with predictions and bounds on unseen test deliveries. An interval crossing 48 hours suggests the delivery could plausibly fall on either side of the deadline, so it is a candidate for review. An interval entirely above 48 hours flags a different issue: a likely late delivery. Neither rule gives conditional coverage for its selected subset or guarantees a service-level success rate.

## Next steps

Choose a policy on development data and evaluate it once on untouched test data. Investigate group coverage and drift before deployment; see [diagnostics](../user-guide/diagnostics.md).

The [statistical specification](../legacy/THEORY.md#evaluation-and-limits) explains the coverage assumptions and limits behind this example.
