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
from src.conformity.regression import ConformalRegressor
from src.conformity.classifier import ConformalClassifier
```

### Example: Conformal Regressor
```python
from sklearn.linear_model import LinearRegression
from src.conformity.regression import ConformalRegressor
import numpy as np

# Generate synthetic data
X_train = np.random.rand(100, 1)
y_train = 3 * X_train.squeeze() + np.random.randn(100) * 0.5
X_test = np.random.rand(10, 1)

# Initialize and fit the regressor
regressor = ConformalRegressor(estimator=LinearRegression())
regressor.fit(X_train, y_train)

# Calibrate the model
regressor.calibrate(X_train, y_train)

# Make predictions with prediction intervals
y_pred, intervals, q_level = regressor.predict(X_test, alpha=0.1)
print("Predictions:", y_pred)
print("Prediction Intervals:", intervals)
```

### Example: Conformal Classifier
```python
from sklearn.ensemble import RandomForestClassifier
from src.conformity.classifier import ConformalClassifier
import numpy as np

# Generate synthetic data
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 3, size=100)
X_test = np.random.rand(10, 5)

# Initialize and fit the classifier
classifier = ConformalClassifier(estimator=RandomForestClassifier())
classifier.fit(X_train, y_train)

# Calibrate the model
classifier.calibrate(X_train, y_train)

# Make predictions with prediction sets
pred_set, boolean_set, y_prob, q_level = classifier.predict(X_test, alpha=0.1)
print("Prediction Sets:", pred_set)
print("Boolean Set:", boolean_set)
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
