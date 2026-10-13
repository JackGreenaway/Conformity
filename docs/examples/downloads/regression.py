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
