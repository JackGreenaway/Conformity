# Conformity

Conformity is a Python library designed to provide conformal prediction tools for regression and classification tasks. It ensures reliable uncertainty quantification in machine learning models.

## Prerequisites

Before using this repository, ensure you have the following installed:
- [Python 3.11](https://www.python.org/downloads/)
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
n_samples = 60

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
y_pred, intervals, q_level = regressor.predict(X_test, alpha=0.1)

print("Predictions:")
pprint(y_pred.reshape(-1, 1))

print("\nPrediction Intervals:")
pprint(intervals)
```

```
Predictions:
array([[0.60955029],
       [2.61894793],
       [2.04560376],
       [1.93151949],
       [0.76095551],
       [1.22316527],
       [1.1236354 ],
       [0.76920256],
       [1.42267417],
       [1.8192274 ],
       [2.02299972],
       [0.59338968]])

Prediction Intervals:
array([[-0.30307492,  1.5221755 ],
       [ 1.70632272,  3.53157314],
       [ 1.13297854,  2.95822897],
       [ 1.01889428,  2.8441447 ],
       [-0.1516697 ,  1.67358072],
       [ 0.31054006,  2.13579048],
       [ 0.21101019,  2.03626061],
       [-0.14342265,  1.68182777],
       [ 0.51004896,  2.33529938],
       [ 0.90660219,  2.73185262],
       [ 1.11037451,  2.93562493],
       [-0.31923553,  1.50601489]])
```

### Example: Conformal Classifier
```python
import numpy as np
from pprint import pprint
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from conformity.classifier import ConformalClassifier

np.random.seed(345)

# Generate synthetic data
n_samples = 200

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
pred_set, boolean_set, y_prob, q_level = classifier.predict(X_test, alpha=0.1)

print("Prediction Sets:")
pprint(pred_set)

print("\nBoolean Set:")
pprint(boolean_set)
```

```
Prediction Sets:
array([[ 0., nan, nan],
       [nan,  1., nan],
       [nan, nan,  2.],
       [nan,  1., nan],
       [ 0., nan, nan],
       [ 0., nan, nan],
       [ 0., nan, nan],
       [ 0., nan, nan],
       [nan,  1., nan],
       [nan,  1., nan]])

Boolean Set:
array([[ True, False, False],
       [False,  True, False],
       [False, False,  True],
       [False,  True, False],
       [ True, False, False],
       [ True, False, False],
       [ True, False, False],
       [ True, False, False],
       [False,  True, False],
       [False,  True, False]])
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
