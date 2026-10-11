# Estimators and pipelines

## Choosing a component

| Component | When to use it |
| --- | --- |
| `ConformalRegressor` | A single numeric target needs a held-out prediction range without a parametric noise assumption; see [regression guidance](regression.md). |
| `ConformalClassifier` | A categorical decision should expose several plausible labels or trigger review; see [classification guidance](classification.md). |
| `BaseConformalPredictor` | You are extending the calibration lifecycle with a new score; use a concrete wrapper for ordinary prediction. |
| Public metrics and `evaluate` | You need to assess coverage and efficiency on independent observations; see [diagnostics](diagnostics.md). |

## Shared estimator behavior

Constructors retain configuration, `fit` trains a fresh clone, and `set_params` invalidates fitted and calibrated state. Point predictions work after fitting; uncertainty output needs calibration. `score` measures point performance. Fit and evaluation weights do not weight calibration.

## Keep calibration out of learning

Use a conformal wrapper around the **whole fitted preprocessing and prediction pipeline**. The wrapper clones its constructor `estimator`, fits the clone as `estimator_`, and leaves the original unfitted. Dense arrays, CSR/CSC sparse matrices and pandas DataFrames are accepted, subject to the wrapped model's own requirements. DataFrame column names are retained for column-selecting pipelines.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 3))
y = X[:, 0] + rng.normal(size=200)
wrapped = make_pipeline(StandardScaler(), Ridge())
model = ConformalRegressor(wrapped, random_state=0).fit(X, y)
assert not hasattr(wrapped[-1], "coef_")
assert model.estimator_[0].n_samples_seen_ == 160
assert model.n_calibration_ == 40
model.set_params(estimator__ridge__alpha=2.0)
assert not hasattr(model, "estimator_")
assert not model.is_calibrated_
```

## Results and interpretation

The scaler sees only 160 fitting rows; 40 are reserved for residual calibration. `set_params` invalidates the fitted clone and all calibration state, so refit after changing model configuration. Refitting creates a fresh clone. Recalibration replaces scores and warns when scores already exist.

## Workflow choices

Constructor `auto_calibrate=True` supports normal scikit-learn `fit`, cloning and cross-validation. Use `auto_calibrate=False` when you have an explicit independent calibration partition. Fit-time overrides are available but constructor choices are easier to preserve through cloning.

An outer preprocessing step followed by a conformal wrapper is supported computationally, but preprocessing then learns from the internal calibration features. That placement does not satisfy the usual split-conformal proof. See [the preserved pipeline discussion](../legacy/THEORY.md#pipeline-placement-and-selection).

Fit weights and step-qualified `*__sample_weight` vectors are validated and sliced with the fitting split. Calibration remains unweighted. Other fit metadata is forwarded unchanged; callers must handle its alignment. Weighted evaluation describes a weighted test population and does not establish weighted calibration coverage.

## Extending the base class

`BaseConformalPredictor` is abstract: subclasses must implement `calibrate` to compute and store held-out scores. Its shared lifecycle manages the fitted clone, input validation, score storage and finite-sample thresholds. A new scoring method also needs its own prediction-region construction and statistical justification; inheriting the lifecycle alone does not establish weighted or conditional coverage.

## Next steps

Use [cross-validation](../examples/cross-validation.md) for point-model development, then reserve untouched calibration and test data for final uncertainty evaluation. Inspect the [generated class references](../api.md) for signatures and lifecycle methods.
