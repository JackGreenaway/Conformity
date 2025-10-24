# Conformity

Conformity is a Python library designed to provide conformal prediction tools for regression and classification tasks. It ensures reliable uncertainty quantification in machine learning models.

## Prerequisites

Before using this repository, ensure you have the following installed:
- [UV](https://uv.pm/) (for package management)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/conformity.git
   cd conformity
   ```

2. Install dependencies using `UV`:
   ```bash
   uv install
   ```
   or
   ```bash
   uv sync
   ```
   

## Usage

### Running the Project
To use the library, you can import the modules and classes in your Python scripts. For example:
```python
from conformity.regressor import ConformalRegressor
from conformity.classifier import ConformalClassifier
```

### Example: Conformal Regressor
```python
import numpy as np
from pprint import pprint
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from conformity.regressor import ConformalRegressor

np.random.seed(434)

# Generate synthetic data
n_samples = 500

x = np.random.rand(n_samples, 1)
y = 3 * x.squeeze() + np.random.randn(n_samples) * 0.5

# Create train, calib, and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train, y_train, test_size=0.3, shuffle=True
)

# Initialize and fit the regressor
regressor = ConformalRegressor(estimator=LinearRegression())
regressor.fit(X_train, y_train)

# Calibrate the model
regressor.calibrate(X_calib, y_calib)

# Make predictions with prediction intervals
y_pred, intervals, q_level = regressor.predict(X_test, alpha=0.05)

print("Predictions:")
pprint(y_pred.reshape(-1, 1)[:5])

print("\nPrediction Intervals:")
pprint(intervals[:5])
```

```
Predictions:
array([[1.38772709],
       [0.05650726],
       [1.24436876],
       [1.17639525],
       [2.01712604]])

Prediction Intervals:
array([[ 0.46679868,  2.3086555 ],
       [-0.86442115,  0.97743567],
       [ 0.32344035,  2.16529717],
       [ 0.25546684,  2.09732366],
       [ 1.09619763,  2.93805445]])
```

#### Model Validation

```python
from conformity.metrics import (
    prediction_interval_coverage,
    prediction_interval_efficiency,
    prediction_interval_ratio,
    prediction_interval_mse,
)

coverage = prediction_interval_coverage(y_true=y_test, prediction_intervals=intervals)
print(f"Prediction Interval Coverage: {coverage:.03f}")

efficiency = prediction_interval_efficiency(
    point_prediction=y_pred, prediction_intervals=intervals
)
print(f"Prediction Interval Efficiency: {efficiency:.03f}")

ratio = prediction_interval_ratio(
    point_predictions=y_pred, prediction_intervals=intervals
)
print(f"Prediction Interval Ratio: {ratio:.03f}")

mse = prediction_interval_mse(y_true=y_pred, prediction_intervals=intervals)
print(f"Prediction Interval MSE: {mse[0]}, {mse[1]}")
```

```
Prediction Set Coverage: 0.950
Prediction Set Efficiency: 1.842
Prediction Set Ratio: 3.758
Prediction Set MSE: 0.848109134815186, 0.8481091348151861
```

### Example: Conformal Classifier
```python
import numpy as np
from pprint import pprint
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from conformity.classifier import ConformalClassifier

np.random.seed(653)

# Generate synthetic data
n_samples = 500

x, y = make_classification(
    n_samples=n_samples, n_features=5, n_clusters_per_class=1, n_classes=3
)

# Create train, calib, and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, shuffle=True)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train, y_train, test_size=0.3, shuffle=True
)


# Initialize and fit the classifier
classifier = ConformalClassifier(estimator=RandomForestClassifier())
classifier.fit(X_train, y_train)

# Calibrate the model
classifier.calibrate(X_calib, y_calib)

# Make predictions with prediction sets
pred_set, boolean_set, y_prob, q_level = classifier.predict(X_test, alpha=0.05)

print("Prediction Sets:")
pprint(pred_set[:5])

print("\nBoolean Set:")
pprint(boolean_set[:5])
```

```
Prediction Sets:
array([[nan, nan,  2.],
       [nan, nan,  2.],
       [nan, nan,  2.],
       [nan, nan,  2.],
       [ 0., nan, nan]])

Boolean Set:
array([[False, False,  True],
       [False, False,  True],
       [False, False,  True],
       [False, False,  True],
       [ True, False, False]])
```

#### Model Validation

```python
from conformity.metrics import prediction_set_coverage, prediction_set_efficiency

coverage = prediction_set_coverage(y_true=y_test, prediction_set=pred_set)
print(f"Prediction Set Coverage: {coverage:.03f}")

efficiency = prediction_set_efficiency(prediction_set=pred_set)
print(f"Prediction Set Coverage: {efficiency:.03f}")
```

```
Prediction Set Coverage: 0.960
Prediction Set Efficiency: 0.000
```


### Development
For development purposes, install the `dev` dependencies:
```bash
uv install --group dev
```

### Building the Project
To build the project for distribution:
```bash
uv build
```

## Features

- **Conformal Regressor**: Provides prediction intervals for regression tasks.
- **Conformal Classifier**: Generates prediction sets for classification tasks.
- **Metrics**: Includes utilities to evaluate coverage and efficiency of prediction intervals and sets.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Open a pull request.
