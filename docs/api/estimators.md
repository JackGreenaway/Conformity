# Estimators

Conformity wraps scikit-learn estimators to add split-conformal calibration.
The wrapper clones and fits the estimator, then uses held-out calibration data
to produce prediction intervals or sets.

| Estimator | Purpose | Reference |
| --- | --- | --- |
| `BaseConformalPredictor` | Shared lifecycle and interface for conformal estimators | [Reference](../classes/base-conformal-predictor.md) |
| `ConformalRegressor` | Prediction intervals for numeric targets | [Reference](../classes/conformal-regressor.md) |
| `ConformalClassifier` | Prediction sets for class labels | [Reference](../classes/conformal-classifier.md) |

See [estimator and pipeline guidance](../user-guide/estimators.md) for how fitting,
preprocessing and calibration interact. Browse the [metrics reference](metrics.md)
for evaluation helpers.
