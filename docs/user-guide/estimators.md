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

## Final-step wrapper and manual calibration

Normally put the conformal wrapper on the final model, hold out calibration data before `Pipeline.fit`, and call `model[-1].calibrate(model[:-1].transform(X_calib), y_calib)`. See the [getting started example](../getting-started.md).

### Why the split matters

Calibration measures errors of an already fixed prediction procedure. That procedure includes learned preprocessing as well as the model: `StandardScaler`, for example, estimates means and variances from the rows passed to `fit`. Hold calibration rows out of all those fitting steps so their scores and a future observation's score can be treated exchangeably under the usual split-conformal assumptions. Keeping only calibration **targets** out of fitting is insufficient for this argument if preprocessing still learns from calibration **features**.

With the wrapper on the final model, the sequence is: split training, calibration and test data; fit the pipeline on training data; transform calibration features with the fitted prefix; calibrate the final wrapper; predict through the full pipeline. This layout makes preprocessing ordinary named sklearn steps and gives you explicit control over the calibration partition. Use `auto_calibrate=False` because you have already made that partition; otherwise the wrapper also reserves part of the training split internally.

| Situation | Placement and calibration | Why |
| --- | --- | --- |
| You can reserve calibration before fitting | `Pipeline` ending in a wrapped model, with manual calibration | Every learned step fits on training rows only, and you control the held-out partition. |
| You need ordinary `fit` to create calibration automatically, including inside cross-validation folds | Wrap the entire `Pipeline`, with `auto_calibrate=True` | The wrapper splits before fitting any learned step. A final-step wrapper cannot control preceding transformers. |
| Preceding transformations are stateless or fixed independently of calibration | A final-step wrapper can use automatic calibration | Those transformations learn nothing from the internal calibration rows. Future inputs must receive the same fixed transformation. |
| You only need point predictions during development | Either placement supports sklearn point scoring | Successful point scoring does not establish a conformal coverage guarantee; reserve fresh calibration for the selected procedure. |

Both placements can preserve the usual marginal coverage guarantee when used with the appropriate split and assumptions. Placement alone is not enough: neither layout fixes distribution shift, tuning on final calibration outcomes, or reusing training rows as calibration.

### Calling methods on the final step

`Pipeline.predict` passes inputs through preprocessing and forwards prediction options to the final wrapper with sklearn metadata routing disabled. Standard `Pipeline` does not expose the wrapper's custom `calibrate`, `evaluate`, `predict_interval`, or `predict_set` methods. Call these on `model.named_steps["conformal"]` after transforming inputs with `model[:-1].transform(...)`. Passing raw features directly to that wrapper would bypass preprocessing. For a wrapper around the whole pipeline, these methods accept raw features because its wrapped estimator performs the transformations.

## Whole-pipeline wrapper and automatic calibration

The example below deliberately wraps the whole pipeline to demonstrate **automatic calibration isolation**: the internal split must happen before the scaler fits. The wrapper clones its constructor `estimator`, fits the clone as `estimator_`, and leaves the original unfitted. Dense arrays, CSR/CSC sparse matrices and pandas DataFrames are accepted, subject to the wrapped model's own requirements. DataFrame column names are retained for column-selecting pipelines.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 3))
y = X[:, 0] + rng.normal(size=200)
wrapped = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
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

An outer **learned** preprocessing step followed by an **automatically calibrating** wrapper is supported computationally, but preprocessing then learns from the internal calibration features. That combination does not satisfy the usual split-conformal proof. Manual calibration held out before the entire pipeline fit avoids this issue; stateless or independently fixed transformations also avoid learning from the internal split. See [the preserved pipeline discussion](../legacy/THEORY.md#pipeline-placement-and-selection).

Fit weights and step-qualified `*__sample_weight` vectors are validated and sliced with the fitting split. Calibration remains unweighted. Other fit metadata is forwarded unchanged; callers must handle its alignment. Weighted evaluation describes a weighted test population and does not establish weighted calibration coverage.

## Extending the base class

`BaseConformalPredictor` is abstract: subclasses must implement `calibrate` to compute and store held-out scores. Its shared lifecycle manages the fitted clone, input validation, score storage and finite-sample thresholds. A new scoring method also needs its own prediction-region construction and statistical justification; inheriting the lifecycle alone does not establish weighted or conditional coverage.

## Next steps

Use [cross-validation](../examples/cross-validation.md) for point-model development, then reserve untouched calibration and test data for final uncertainty evaluation. Inspect the [generated class references](../api.md) for signatures and lifecycle methods.
