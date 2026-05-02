# Conformity - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Examples & Tutorials](#examples--tutorials)
6. [Performance Considerations](#performance-considerations)
7. [Troubleshooting](#troubleshooting)

## Overview

Conformity is a scikit-learn compatible library implementing conformal prediction, a powerful framework for uncertainty quantification. It provides **valid statistical guarantees** on prediction coverage without making distributional assumptions about the data.

### What is Conformal Prediction?

Conformal prediction is a framework that:
- Complements any base predictor (linear models, neural networks, random forests, etc.)
- Produces prediction sets/intervals with **guaranteed coverage** under exchangeability
- Makes **no distributional assumptions** about the data
- Works with finite samples (no asymptotic theory required)

### Key Advantages

1. **Distribution-Free**: No parametric assumptions; works with any data distribution
2. **Valid Coverage**: Marginal coverage guarantees regardless of the base model's quality
3. **Model-Agnostic**: Works with any scikit-learn compatible estimator
4. **Computationally Efficient**: $O(n \log n)$ complexity due to quantile computation
5. **Scikit-Learn Integration**: Full compatibility with pipelines, cross-validation, etc.

## Installation & Setup

### Requirements

- Python 3.9+
- numpy >= 2.0.2
- scikit-learn >= 1.6.1
- typing-extensions >= 4.15.0

### Quick Install

```bash
pip install conformity-calib
```

### Development Installation

```bash
git clone https://github.com/your-username/conformity.git
cd conformity
uv sync
uv run pytest tests/
```

## Core Concepts

### The Three-Set Split

Conformal prediction requires data to be split into three independent sets:

```
Original Data (n samples)
    ├── Training Set (50%): Used to train the base estimator
    ├── Calibration Set (25%): Used to compute non-conformity scores
    └── Test Set (25%): For evaluation
```

**Important**: These sets must be independent. Reusing training data for calibration violates the exchangeability assumption.

### Non-Conformity Scores

Non-conformity scores measure how "unusual" a prediction is:

- **Regression**: $\alpha_i = |y_i - \hat{y}_i|$ (absolute residual)
- **Classification**: $\alpha_i = 1 - P(Y_i | X_i)$ (1 minus the true class probability)

Higher scores indicate more unusual/uncertain predictions.

### The Prediction Mechanism

1. Compute non-conformity scores on calibration set
2. For a new sample, compute its non-conformity score
3. Compare against calibration scores to determine if it conforms
4. Construct prediction interval/set based on conformity assessment

### Coverage Guarantee

For any significance level $\alpha \in (0,1)$:

$$P(\text{true value} \in \text{prediction set}) \geq 1 - \alpha - \frac{1}{n+1}$$

where $n$ is the calibration set size. This holds for any data distribution!

## API Reference

### BaseConformalPredictor

Abstract base class for all conformal predictors.

```python
class BaseConformalPredictor(BaseEstimator, ABC):
    """
    Parameters
    ----------
    estimator : BaseEstimator
        Base predictor to wrap
    
    Attributes
    ----------
    estimator_ : BaseEstimator
        Fitted estimator
    is_calibrated_ : bool
        Whether calibration has been performed
    """
```

**Methods:**

- `fit(X, y, auto_calibrate=False, tts_kwargs=None)` → self
  - Fit base estimator
  - If `auto_calibrate=True`, automatically split and calibrate
  - Returns self for method chaining

- `calibrate(X_calib, y_calib)` → self
  - Compute non-conformity scores on calibration data
  - Must call `fit()` first
  - Returns self for method chaining

### ConformalRegressor

Conformal predictor for regression with prediction intervals.

```python
from conformity.regressor import ConformalRegressor
from sklearn.linear_model import LinearRegression

reg = ConformalRegressor(estimator=LinearRegression())
reg.fit(X_train, y_train)
reg.calibrate(X_calib, y_calib)
y_pred, intervals, q_level = reg.predict(X_test, alpha=0.1)
```

**Parameters:**
- `estimator` : RegressorMixin
  - Any scikit-learn compatible regression estimator

**Attributes:**
- `estimator_` : Fitted estimator
- `is_calibrated_` : bool
- `calibration_non_conformity` : array (n_calib,)
  - Absolute residuals from calibration set
- `n_calib` : int
  - Number of calibration samples

**Methods:**
- `fit(X, y, auto_calibrate=False, tts_kwargs=None)` → self
- `calibrate(X_calib, y_calib)` → self
- `predict(X, alpha=0.05)` → (y_pred, intervals, q_level)
  - `y_pred` : array (n_samples,) - Point predictions
  - `intervals` : array (n_samples, 2) - [lower, upper] bounds
  - `q_level` : float - Quantile level used

### ConformalClassifier

Conformal predictor for classification with prediction sets.

```python
from conformity.classifier import ConformalClassifier
from sklearn.linear_model import LogisticRegression

clf = ConformalClassifier(estimator=LogisticRegression())
clf.fit(X_train, y_train)
clf.calibrate(X_calib, y_calib)
pred_set, bool_set, probs, q_level = clf.predict(X_test, alpha=0.1)
```

**Parameters:**
- `estimator` : ClassifierMixin
  - Any scikit-learn compatible classifier (must have `predict_proba`)

**Attributes:**
- `estimator_` : Fitted estimator
- `is_calibrated_` : bool
- `calibration_non_conformity` : array (n_calib,)
  - 1 - true_class_probability for calibration set
- `n_calib` : int
  - Number of calibration samples

**Methods:**
- `fit(X, y, auto_calibrate=False, tts_kwargs=None)` → self
- `calibrate(X_calib, y_calib)` → self
- `predict(X, alpha=0.05)` → (pred_set, bool_set, probs, q_level)
  - `pred_set` : array (n_samples, n_classes) - Prediction sets (NaN = not in set)
  - `bool_set` : array (n_samples, n_classes) - Boolean membership
  - `probs` : array (n_samples, n_classes) - Class probabilities
  - `q_level` : float - Quantile level used

### Metrics Module

```python
from conformity import metrics

# Regression metrics
coverage = metrics.prediction_interval_coverage(y_true, intervals)
efficiency = metrics.prediction_interval_efficiency(y_pred, intervals)
mse_lower, mse_upper = metrics.prediction_interval_mse(y_true, intervals)
ratio = metrics.prediction_interval_ratio(y_pred, intervals)

# Classification metrics
coverage = metrics.prediction_set_coverage(y_true, pred_set)
efficiency = metrics.prediction_set_efficiency(pred_set)
```

**Regression Metrics:**

- `prediction_interval_coverage(y_true, intervals)` → float
  - Fraction of samples where true value is in interval
  - Range: [0, 1], Target: ≥ 1-α

- `prediction_interval_efficiency(y_pred, intervals, relative=False)` → float
  - Average interval width
  - Lower is better (narrower intervals)
  - If `relative=True`, normalize by prediction magnitude

- `prediction_interval_mse(y_true, intervals)` → (float, float)
  - MSE of lower and upper bounds
  - Returns (mse_lower, mse_upper)

- `prediction_interval_ratio(y_pred, intervals)` → float
  - Mean ratio of upper bound to predictions
  - Useful for comparing interval shapes

**Classification Metrics:**

- `prediction_set_coverage(y_true, pred_set)` → float
  - Fraction of samples where true class is in set
  - Range: [0, 1], Target: ≥ 1-α

- `prediction_set_efficiency(pred_set)` → float
  - Average set size (normalized 0-1)
  - Lower is better (smaller sets)

## Examples & Tutorials

### Example 1: Basic Regression

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from conformity.regressor import ConformalRegressor
from conformity.metrics import prediction_interval_coverage

# Generate data
X, y = make_regression(n_samples=500, n_features=10, random_state=42)

# Split into train, calib, test
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

# Fit conformal regressor
reg = ConformalRegressor(LinearRegression())
reg.fit(X_train, y_train)
reg.calibrate(X_calib, y_calib)

# Predict with intervals
y_pred, intervals, q_level = reg.predict(X_test, alpha=0.1)

# Evaluate
coverage = prediction_interval_coverage(y_test, intervals)
print(f"Coverage: {coverage:.3f} (target ≥ 0.9)")
```

### Example 2: Classification with Different Alpha Levels

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from conformity.classifier import ConformalClassifier
from conformity.metrics import prediction_set_coverage, prediction_set_efficiency

X, y = make_classification(n_samples=500, n_features=10, random_state=42)

X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

clf = ConformalClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)
clf.calibrate(X_calib, y_calib)

# Test different alpha levels
for alpha in [0.01, 0.05, 0.1, 0.2]:
    pred_set, _, _, _ = clf.predict(X_test, alpha=alpha)
    coverage = prediction_set_coverage(y_test, pred_set)
    efficiency = prediction_set_efficiency(pred_set)
    print(f"α={alpha}: Coverage={coverage:.3f}, Efficiency={efficiency:.3f}")
```

### Example 3: Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from conformity.regressor import ConformalRegressor

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ConformalRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
])

# Use with cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(
    pipeline, X, y, cv=5, 
    scoring='neg_mean_squared_error'
)
print(f"CV Scores: {scores}")
```

### Example 4: Auto-Calibration

```python
from conformity.regressor import ConformalRegressor
from sklearn.linear_model import LinearRegression

# Fit and calibrate in one step
reg = ConformalRegressor(LinearRegression())
reg.fit(
    X, y,  # Full dataset
    auto_calibrate=True,
    tts_kwargs={'test_size': 0.3, 'random_state': 42}
)

# Ready to predict
y_pred, intervals, _ = reg.predict(X_new, alpha=0.1)
```

## Performance Considerations

### Time Complexity

- `fit()`: $O(n \cdot \text{base_estimator_time})$
- `calibrate()`: $O(n \log n)$ (for quantile computation)
- `predict()`: $O(m \cdot n)$ where $m$ is number of test samples

### Memory Complexity

- Stores calibration non-conformity scores: $O(n_{\text{calib}})$
- No other significant overhead

### Optimization Tips

1. **Larger calibration sets improve coverage** but require more computation
   - Typical: 20-30% of training data

2. **Use efficient base estimators** for faster prediction
   - Linear models: Very fast
   - Trees/Forests: Moderate
   - Neural networks: Can be slow

3. **Consider data preprocessing** before conformal wrapping
   - StandardScaler in pipeline is compatible

4. **Batch predictions** if predicting many samples
   - All predictions computed together

## Troubleshooting

### Issue: "Must call fit before calibrate"

**Cause**: Calling `calibrate()` without first calling `fit()`

**Solution**:
```python
reg = ConformalRegressor(LinearRegression())
reg.fit(X_train, y_train)  # DO THIS FIRST
reg.calibrate(X_calib, y_calib)
```

### Issue: Coverage much lower than expected

**Common Causes**:
1. Too small calibration set (n < 30 recommended)
2. Calibration set not independent from training
3. Base estimator not well-calibrated
4. Data violates exchangeability assumption

**Solutions**:
- Increase calibration set size
- Ensure proper data splitting
- Try different base estimators
- Check for temporal structure in data

### Issue: Very wide prediction intervals

**Causes**:
1. Base estimator makes poor predictions
2. High noise in data
3. Small alpha (e.g., alpha=0.01)

**Solutions**:
- Try better base estimator
- Check data quality
- Increase alpha (less stringent coverage)

### Issue: Pipeline compatibility errors

**Solution**: Ensure your base estimator is scikit-learn compatible:
```python
from sklearn.utils.validation import check_is_fitted

# Should have these methods:
estimator.fit(X, y)
estimator.predict(X)  # For regressor
estimator.predict_proba(X)  # For classifier
```

### Issue: Memory error on large datasets

**Solution**: Process in batches:
```python
# Train on subset if needed
sample_indices = np.random.choice(len(X), size=5000, replace=False)
reg.fit(X[sample_indices], y[sample_indices])
reg.calibrate(X_calib, y_calib)

# Predict in batches
batch_size = 1000
predictions = []
for i in range(0, len(X_new), batch_size):
    batch = X_new[i:i+batch_size]
    y_pred, intervals, _ = reg.predict(batch)
    predictions.append((y_pred, intervals))
```

## FAQ

**Q: Do I need labeled data for calibration?**
A: Yes, calibration requires true labels to compute non-conformity scores.

**Q: Can I use the same data for training and calibration?**
A: No, this violates the exchangeability assumption and invalidates coverage guarantees.

**Q: What if my intervals are too wide?**
A: This usually indicates the base model has high residuals. Try more sophisticated models.

**Q: How do I choose alpha?**
A: Alpha controls the desired coverage level. α=0.1 means 90% coverage. Choose based on your application's requirements.

**Q: Can this work with neural networks?**
A: Yes! Any scikit-learn compatible predictor works, including sklearn wrappers for neural networks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## References

1. Vovk, V., Gammerman, A., & Shafer, G. (1999). "Algorithmic Learning Theory"
2. Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2019). "Predictive inference with the jackknife+"
3. Angelopoulos, A. N., & Bates, S. (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
