# API reference

The [README](README.md) contains workflows. [THEORY.md](THEORY.md) specifies the statistical assumptions, scores, rank proof and limits.

## Default workflow

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from conformity import ConformalRegressor

X, y = make_regression(n_samples=1000, noise=20, random_state=42)
X_new = X[:5]  # Replace with new observations in practice.

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("conformal", ConformalRegressor(
        Ridge(),
        prediction_mode="point",
        auto_calibrate=True,
        random_state=42,
    )),
])

results = cross_validate(pipe, X, y, cv=5, scoring="r2")

pipe.fit(X, y)
points = pipe.predict(X_new)
points, intervals = pipe.predict(X_new, return_interval=True)
```

This is the default usage pattern in these docs. `Pipeline` is imported directly from `sklearn.pipeline` and takes a list of named steps. Point mode works with ordinary sklearn scorers; request intervals explicitly with `return_interval=True` (optionally pass `alpha`, which defaults to `0.05`). `cross_validate` fits clones, so fit the original pipeline before predicting.

The outer scaler fits on all observations passed to each pipeline fit, including the final wrapper's internal calibration features. This workflow is supported, but the usual split-conformal coverage proof does not apply to that placement of learned preprocessing. For that guarantee, put learned preprocessing inside the wrapped estimator, as described in [THEORY.md](THEORY.md#pipeline-placement-and-selection).

## Lifecycle

1. Construct `ConformalRegressor(estimator, *, auto_calibrate=False, tts_kwargs=None, calibration_size=0.2, random_state=None, prediction_mode="conformal")` or `ConformalClassifier(estimator, *, method="lac", auto_calibrate=False, tts_kwargs=None, calibration_size=0.2, random_state=None, prediction_mode="conformal")`.
2. `fit(X, y, auto_calibrate=None, tts_kwargs=None, *, sample_weight=None, **fit_params)` validates inputs, discards stale fitted/calibration state, clones the estimator, and fits it. With automatic calibration, splitting happens before the wrapped estimator is fitted. Constructor defaults are merged with `tts_kwargs`. Unqualified and step-qualified `*__sample_weight` vectors are validated and sliced to the training partition. Other metadata is forwarded unchanged; unqualified weights are not automatically routed into a pipeline.
3. `calibrate(X, y)` validates the feature schema and targets and replaces calibration scores. Invalid calibration inputs leave previous valid scores intact. Calibration data must be independent of fitting and tuning.
4. `predict_point(X)` and `score(X, y, sample_weight=None)` need fitting only. Classification also exposes `predict_proba(X)`.
5. `predict_interval(X, alpha=0.05)` or `predict_set(X, alpha=0.05)` needs calibration. Alpha must be a finite scalar strictly between zero and one. Boolean sets use `classes_` order.
6. `predict(X, alpha=0.05)` returns ordinary predictions with the recommended `prediction_mode="point"`. Use `return_interval=True` for regression or `return_set=True` for classification to request a conformal tuple explicitly. Legacy conformal mode returns that tuple by default. Alpha does not affect point predictions.
7. `evaluate(X, y, alpha=0.05, *, sample_weight=None)` returns a dictionary of point and conformal diagnostics.

`fit` and `calibrate` return `self`. Calling `set_params`, including nested estimator parameters, invalidates both the fitted estimator and calibration. Refitting invalidates calibration even if the refit fails. Recalibration warns before replacing existing scores. Changing a classifier's `method` directly requires recalibration; use `set_params` for normal sklearn configuration.

## Fitted attributes

| Attribute                             | Meaning                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `estimator_`                          | Fitted clone; the supplied `estimator` is untouched                       |
| `n_features_in_`, `feature_names_in_` | Validated input schema; names when provided as string DataFrame columns   |
| `classes_`                            | Fitted classification labels in probability-column order                  |
| `is_calibrated_`                      | Read-only-in-normal-use calibration status; false before calibration      |
| `calibration_scores_`                 | Immutable copy of calibration nonconformity scores                        |
| `sorted_calibration_scores_`          | Immutable sorted cache for thresholds and p-values                        |
| `n_calibration_`                      | Number of calibration observations                                        |
| `alpha_used_`                         | Most recent conformal prediction's alpha                                  |
| `q_level_`                            | Most recent score threshold, possibly infinity                            |
| `quantile_level_`                     | Finite-sample rank divided by calibration size, possibly greater than one |
| `method_`                             | Classifier score method used for the current calibration                  |

`calibration_non_conformity` and `n_calib` remain compatibility aliases. `q_level_` consistently means a score threshold for both tasks; older classifier versions used a probability level under this name.

Prediction records the most recent alpha/threshold for introspection. Concurrent prediction calls can overwrite these diagnostics; use the returned arrays as the result of each call. Do not concurrently mutate, fit, or recalibrate one instance. Clone separate instances for independent fitting. The wrapped estimator must itself be safe for concurrent inference if inference is shared.

## Metrics

Coverage includes endpoints. Set coverage supports legacy label arrays and boolean membership arrays; boolean arrays require `classes`. Set size, empty rate, and singleton rate work with either representation. All diagnostics aggregate over samples and accept keyword-only `sample_weight`. Weights must be finite, nonnegative, correctly shaped, and have positive total weight.

Interval inputs must have nonempty shape `(n_samples, 2)` with ordered bounds and no NaN. Outward infinite bounds are accepted and have infinite width/interval score. Zero-weight observations are excluded before weighted aggregation, so an unbounded zero-weight interval does not create `0 * infinity` NaNs.

`interval_score` is width plus `2 / alpha` times the distance below the lower bound or above the upper bound. Use the same alpha as the interval construction. The legacy relative-width diagnostic divides by `abs(point_prediction) + 1e-10`; near-zero points therefore yield very large relative widths. Bound MSE and upper/point ratio are descriptive diagnostics, not coverage guarantees or proper point-prediction losses.

## Scope

This release implements unweighted split conformal absolute-residual regression and LAC/deterministic APS classification. It does not implement CV+, jackknife+, conformalized quantile regression, conditional/Mondrian calibration, online/time-series guarantees, weighted covariate-shift calibration, or randomized scores. Adding these needs separate calibration protocols and dedicated statistical validation.

Sklearn integration uses mixins before `BaseEstimator`, estimator cloning, validation, and fitted attributes. Point mode is the standard estimator contract; legacy conformal mode intentionally has a tuple prediction API and cannot be passed to ordinary prediction-based sklearn scorers. The default workflow uses the wrapper as the final pipeline step. For the usual split-conformal coverage guarantee, put learned preprocessing inside the supplied estimator so calibration observations are excluded before it fits.

Constructor `auto_calibrate` (default False) and `tts_kwargs` configure ordinary fit calls. Explicit fit-time calibration flags override the constructor. Regression `predict(..., return_interval=True/False)` and classification `predict(..., return_set=True/False)` override prediction mode per call and support prediction through a standard Pipeline with metadata routing disabled.

## Constructor parameters

Both wrappers require `estimator` as their only positional parameter. All other parameters are keyword-only. The estimator must be sklearn-cloneable and expose `fit` and `predict`; classifiers additionally require `predict_proba` and fitted `classes_`. The supplied instance is never fitted directly.

| Parameter                  | Default       | Behavior                                                                                  |
| -------------------------- | ------------- | ----------------------------------------------------------------------------------------- |
| `estimator`                | Required      | Model, complete Pipeline, or search estimator to clone and fit                            |
| `auto_calibrate`           | `False`       | Reserve calibration observations during ordinary `fit`                                    |
| `tts_kwargs`               | `None`        | `train_test_split` options; overrides `calibration_size` and `random_state` when supplied |
| `calibration_size`         | `0.2`         | Calibration fraction or count, using sklearn `test_size` semantics                        |
| `random_state`             | `None`        | Split seed; does not change the underlying model's seed                                   |
| `prediction_mode`          | `"conformal"` | Tuple predictions by default; `"point"` for sklearn prediction-based tools                |
| `method` (classifier only) | `"lac"`       | `"lac"` or deterministic cumulative-score `"aps"`                                         |

At fit time, `auto_calibrate=None` uses the constructor value. Explicit `False` disables automatic calibration for that fit. `tts_kwargs=None` uses constructor split options when automatic calibration is enabled; an explicit dictionary replaces those options. Supplying fit-time split options while automatic calibration is disabled raises `ValueError`.

## Prediction methods and shapes

Here `m` is the number of query rows and `c` is the number of fitted classes.

| Method                                     | Regressor                         | Classifier                               |
| ------------------------------------------ | --------------------------------- | ---------------------------------------- |
| `predict_point(X)`                         | Float vector `(m,)`               | Label vector `(m,)`                      |
| `predict(X, alpha=0.05)` in conformal mode | `(points, bounds)`                | `(legacy_label_sets, probabilities)`     |
| `predict(X, alpha=0.05)` in point mode     | Same as `predict_point`           | Same as `predict_point`                  |
| `predict_interval(X, alpha=0.05)`          | Bounds `(m, 2)`, lower then upper | Unavailable                              |
| `predict_set(X, alpha=0.05)`               | Unavailable                       | Boolean membership `(m, c)`              |
| `predict_proba(X)`                         | Unavailable                       | Probabilities `(m, c)`                   |
| `predict_p_values(X)`                      | Unavailable                       | Conservative conformal p-values `(m, c)` |
| `score(X, y, sample_weight=None)`          | Point R²                          | Point accuracy                           |

Regression `return_interval` and classification `return_set` accept `True`, `False`, or `None`. `True` forces a conformal tuple, `False` forces points, and `None` follows the constructor mode. Legacy label arrays have shape `(m, c)` with NaN exclusions; boolean membership is preferable for arbitrary label types. All conformal methods require calibration; point methods require fitting only.

`evaluate` returns regression keys `r2`, `mae`, `mse`, `coverage`, `mean_width`, `interval_score`, or classification keys `accuracy`, `log_loss`, `coverage`, `mean_size`, `empty_rate`, `singleton_rate`. Evaluation data should be independent of training, calibration, and model selection when assessing generalization.

## Verification

The suite checks sklearn estimator compliance in point mode, cloning, model selection, internal search isolation, outer pipeline forwarding for both tasks, training-weight alignment, score ties, p-values, and the infinite sentinel. Exact finite-population tests enumerate every possible future index, verifying the coverage lower bound for regression, LAC and deterministic APS, with ties and without relying on simulation tolerances. Tests verify implementation under specified assumptions; they cannot establish exchangeability of a user's data.

## Cross-validation lifecycle

`cross_validate` clones the outer pipeline and wrapper for each fold. Set constructor `auto_calibrate=True` to make its ordinary `fit(X_fold_train, y_fold_train)` perform a fresh internal calibration split. Use `prediction_mode="point"` with standard prediction-based scorers. `return_estimator=True` returns the fitted pipeline clones, each with its own final wrapper's `estimator_`, calibration scores and `n_calibration_`. The original pipeline remains unfitted. `error_score="raise"` surfaces fitting or scoring errors directly. See the [complete cross-validation example](README.md#automatic-calibration-with-cross_validate).

A custom scorer has signature `(fitted_estimator, X_test, y_test)` and returns a scalar. With the default pipeline layout, `fitted_estimator[-1].evaluate(fitted_estimator[:-1].transform(X_test), y_test, alpha=0.1)["coverage"]` scores fold coverage. Each calibration partition comes only from the fold's training data. For the usual coverage proof, learned preprocessing belongs inside the wrapped estimator, so the split excludes calibration observations before any learned transformations fit. A wrapper that is the final outer step cannot move the split ahead of preceding steps.
