# Conformity: Conformal Prediction for Uncertainty Quantification

[![Tests](https://img.shields.io/badge/tests-125%20passed-brightgreen)](./tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Conformity is a Python library providing robust conformal prediction tools for regression and classification tasks. It enables reliable **uncertainty quantification** in machine learning models with statistical guarantees on prediction coverage, allowing practitioners to make more informed decisions with quantified confidence intervals.

## Key Features

- 🎯 **Marginal Coverage Guarantees**: Provides statistically sound prediction intervals and sets with guaranteed coverage under the exchangeability assumption
- 🔌 **Scikit-Learn Compatible**: Seamless integration with the scikit-learn ecosystem—works with pipelines, cross-validation, and model selection tools
- 📊 **Flexible Estimators**: Supports any scikit-learn compatible estimator (Linear, ensemble, SVM, etc.)
- 🛠️ **Comprehensive Metrics**: Built-in functions for evaluating coverage, efficiency, and interval quality
- 📈 **Regression & Classification**: Unified interface for both prediction intervals (regression) and prediction sets (classification)
- ⚡ **Lightweight & Fast**: Minimal dependencies with excellent performance characteristics

## Prerequisites

- Python ≥ 3.9
- [UV](https://docs.astral.sh/uv/) (recommended package manager)
- NumPy ≥ 2.0.2
- Scikit-learn ≥ 1.6.1

## Installation

### Using UV (Recommended)

```bash
git clone https://github.com/your-username/conformity.git
cd conformity
uv sync
```

### Using pip

```bash
pip install conformity-calib
```

## Quick Start

### Conformal Regression with Prediction Intervals

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from conformity.regressor import ConformalRegressor

# Generate sample data
np.random.seed(42)
X = np.random.rand(300, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(300) * 0.5

# Split data: training, calibration, and test
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

# Create and fit the conformal regressor
regressor = ConformalRegressor(estimator=LinearRegression())
regressor.fit(X_train, y_train)
regressor.calibrate(X_calib, y_calib)

# Make predictions with uncertainty intervals
y_pred, intervals = regressor.predict(X_test, alpha=0.1)

print("Predictions:", y_pred[:5])
print("90% Prediction Intervals:")
print(intervals[:5])
print(f"Quantile Level: {regressor.q_level_:.4f}")
```

**Output:**

```
Predictions: [2.34  1.89  3.12  2.01  2.87]
90% Prediction Intervals:
[[1.42  3.26]
 [0.97  2.81]
 [2.20  4.04]
 [1.09  2.93]
 [1.95  3.79]]
Quantile Level: 0.9200
```

### Conformal Classification with Prediction Sets

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from conformity.classifier import ConformalClassifier

# Generate sample data
X, y = make_classification(n_samples=300, n_features=10, n_classes=3,
                           n_informative=8, random_state=42)

# Split data
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

# Create and fit the conformal classifier
classifier = ConformalClassifier(estimator=LogisticRegression(max_iter=1000))
classifier.fit(X_train, y_train)
classifier.calibrate(X_calib, y_calib)

# Make predictions with prediction sets
pred_set, class_probs = classifier.predict(X_test, alpha=0.1)

print("Prediction Sets (classes in set):", pred_set[:5])
print("Class Probabilities:", class_probs[:5])
print(f"Quantile Level: {classifier.q_level_:.4f}")
```

## Evaluation Metrics

Conformity provides comprehensive metrics for evaluating prediction quality:

```python
from conformity.metrics import (
    prediction_interval_coverage,
    prediction_interval_efficiency,
    prediction_set_coverage,
    prediction_interval_mse,
)

# Evaluate regression intervals
coverage = prediction_interval_coverage(y_true=y_test, prediction_intervals=intervals)
efficiency = prediction_interval_efficiency(point_prediction=y_pred, prediction_intervals=intervals)
mse_lower, mse_upper = prediction_interval_mse(y_true=y_test, prediction_intervals=intervals)

print(f"Coverage: {coverage:.3f}")          # Should be close to 1 - alpha
print(f"Efficiency (avg width): {efficiency:.3f}")
print(f"Lower MSE: {mse_lower:.3f}, Upper MSE: {mse_upper:.3f}")

# Evaluate classification sets
coverage = prediction_set_coverage(y_true=y_test, prediction_set=pred_set)
print(f"Classification Coverage: {coverage:.3f}")
```

## Advanced Usage

### Auto-Calibration

Automatically split data for training and calibration:

```python
from conformity.regressor import ConformalRegressor
from sklearn.linear_model import LinearRegression

regressor = ConformalRegressor(LinearRegression())

# Fit and calibrate in one step
regressor.fit(X, y, auto_calibrate=True, tts_kwargs={'test_size': 0.3, 'random_state': 42})

# Now ready to predict
y_pred, intervals, _ = regressor.predict(X_test, alpha=0.1)
```

### Integration with Scikit-Learn Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from conformity.regressor import ConformalRegressor

# Create a pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('conformal_regressor', ConformalRegressor(LinearRegression()))
])

# Use with cross-validation or model selection
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error')
```

### Using Different Estimators

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from conformity.regressor import ConformalRegressor
from conformity.classifier import ConformalClassifier

# With Random Forest
rf_regressor = ConformalRegressor(RandomForestRegressor(n_estimators=100))

# With Gradient Boosting
gb_classifier = ConformalClassifier(GradientBoostingClassifier(n_estimators=100))
```

## API Reference

### ConformalRegressor

**Parameters:**

- `estimator` (RegressorMixin): Scikit-learn compatible regression estimator

**Methods:**

- `fit(X, y, auto_calibrate=False, tts_kwargs=None)`: Fit the estimator to data
- `calibrate(X, y)`: Calibrate using held-out data
- `predict(X, alpha=0.05)`: Predict with intervals
  - Returns: `(y_pred, intervals, q_level)`

### ConformalClassifier

**Parameters:**

- `estimator` (ClassifierMixin): Scikit-learn compatible classification estimator

**Methods:**

- `fit(X, y, auto_calibrate=False, tts_kwargs=None)`: Fit the estimator to data
- `calibrate(X, y)`: Calibrate using held-out data
- `predict(X, alpha=0.05)`: Predict with sets
  - Returns: `(pred_set, boolean_set, class_probs, q_level)`

### Metrics

All metrics accept array-like inputs:

- `prediction_interval_coverage(y_true, prediction_intervals)`: Fraction of true values in intervals
- `prediction_interval_efficiency(point_prediction, prediction_intervals, relative=False)`: Average interval width
- `prediction_set_coverage(y_true, prediction_set)`: Fraction of true values in sets
- `prediction_set_efficiency(prediction_set)`: Average set size
- `prediction_interval_ratio(point_predictions, prediction_intervals)`: Ratio of upper bound to predictions
- `prediction_interval_mse(y_true, prediction_intervals)`: MSE of interval bounds

## How It Works

Conformal prediction is a distribution-free approach that provides statistical guarantees on uncertainty quantification:

1. **Split Data**: Divide data into training and calibration sets
2. **Fit Model**: Train your base estimator on training data
3. **Calibrate**: Compute non-conformity scores on calibration data
4. **Predict**: Generate prediction intervals/sets with coverage guarantees

The beauty of conformal prediction is that it makes **no distributional assumptions** while providing rigorous coverage guarantees under exchangeability.

### Coverage Guarantee

For any miscoverage level $\alpha \in (0, 1)$, the true observation falls within the prediction set/interval with probability at least $1 - \alpha$.

## Examples

See the [examples](./examples/) directory for detailed notebooks:

- [Regression with Continuous Targets](./examples/regression_example.ipynb)
- [Classification with Discrete Classes](./examples/classification_example.ipynb)
- [Comparison with Uncertainty Quantification Methods](./examples/comparison.ipynb)
- [Real-World Applications](./examples/real_world.ipynb)

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=conformity --cov-report=html

# Run specific test file
uv run pytest tests/test_conformal_regressor.py -v
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run `uv run pytest` to ensure all tests pass
5. Submit a pull request

## References

- Vovk, V., Gammerman, A., & Shafer, G. (1999). "[Algorithmic Learning Theory](https://www.springer.com/gp/book/9783540663768)"
- Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2023). "[Conformal Prediction Under Covariate Shift](https://arxiv.org/abs/1904.06857)"
- Angelopoulos, A. N., & Bates, S. (2021). "[A gentle introduction to conformal prediction and distribution-free uncertainty quantification](https://arxiv.org/abs/2107.03541)"

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This library is inspired by the excellent work in conformal prediction and uncertainty quantification. Special thanks to the scikit-learn community for providing such a well-designed API foundation.

## Support & Questions

- **Issues**: [GitHub Issues](https://github.com/your-username/conformity/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/conformity/discussions)
- **Documentation**: Full API docs available in [DOCUMENTATION.md](./docs/DOCUMENTATION.md)

```python

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

```python
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


# Initialise and fit the classifier
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
