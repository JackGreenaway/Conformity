# API reference

The [README](README.md) contains runnable workflows and the statistical contract.

## Lifecycle

1. Construct `ConformalRegressor(estimator, *, calibration_size=0.2, random_state=None, prediction_mode="conformal")` or `ConformalClassifier(estimator, *, method="lac", calibration_size=0.2, random_state=None, prediction_mode="conformal")`.
2. `fit(X, y, auto_calibrate=False, tts_kwargs=None, *, sample_weight=None, **fit_params)` validates inputs, discards stale fitted/calibration state, clones the estimator, and fits it. With automatic calibration, splitting happens before the wrapped estimator is fitted. Constructor defaults are merged with `tts_kwargs`. Extra metadata is forwarded unchanged except the explicitly named `sample_weight` parameter.
3. `calibrate(X, y)` validates the feature schema and targets and replaces calibration scores. Invalid calibration inputs leave previous valid scores intact. Calibration data must be independent of fitting and tuning.
4. `predict_point(X)` and `score(X, y, sample_weight=None)` need fitting only. Classification also exposes `predict_proba(X)`.
5. `predict_interval(X, alpha=0.05)` or `predict_set(X, alpha=0.05)` needs calibration. Alpha must be a finite scalar strictly between zero and one. Boolean sets use `classes_` order.
6. `predict(X, alpha=0.05)` returns a tuple in legacy conformal mode and ordinary predictions in point mode. In point mode, alpha does not affect point predictions.
7. `evaluate(X, y, alpha=0.05, *, sample_weight=None)` returns a dictionary of point and conformal diagnostics.

`fit` and `calibrate` return `self`. Calling `set_params`, including nested estimator parameters, invalidates both the fitted estimator and calibration. Refitting invalidates calibration even if the refit fails. Recalibration warns before replacing existing scores. Changing a classifier's `method` directly requires recalibration; use `set_params` for normal sklearn configuration.

## Fitted attributes

| Attribute | Meaning |
| --- | --- |
| `estimator_` | Fitted clone; the supplied `estimator` is untouched |
| `n_features_in_`, `feature_names_in_` | Validated input schema; names when provided as string DataFrame columns |
| `classes_` | Fitted classification labels in probability-column order |
| `is_calibrated_` | Read-only-in-normal-use calibration status; false before calibration |
| `calibration_scores_` | Immutable copy of calibration nonconformity scores |
| `sorted_calibration_scores_` | Immutable sorted cache for thresholds and p-values |
| `n_calibration_` | Number of calibration observations |
| `alpha_used_` | Most recent conformal prediction's alpha |
| `q_level_` | Most recent score threshold, possibly infinity |
| `quantile_level_` | Finite-sample rank divided by calibration size, possibly greater than one |
| `method_` | Classifier score method used for the current calibration |

`calibration_non_conformity` and `n_calib` remain compatibility aliases. `q_level_` consistently means a score threshold for both tasks; older classifier versions used a probability level under this name.

Prediction records the most recent alpha/threshold for introspection. Concurrent prediction calls can overwrite these diagnostics; use the returned arrays as the result of each call. Do not concurrently mutate, fit, or recalibrate one instance. Clone separate instances for independent fitting. The wrapped estimator must itself be safe for concurrent inference if inference is shared.

## Metrics

Coverage includes endpoints. Set coverage supports legacy label arrays and boolean membership arrays; boolean arrays require `classes`. Set size, empty rate, and singleton rate work with either representation. All diagnostics aggregate over samples and accept keyword-only `sample_weight`. Weights must be finite, nonnegative, correctly shaped, and have positive total weight.

Interval inputs must have nonempty shape `(n_samples, 2)` with ordered bounds and no NaN. Outward infinite bounds are accepted and have infinite width/interval score. Zero-weight observations are excluded before weighted aggregation, so an unbounded zero-weight interval does not create `0 * infinity` NaNs.

`interval_score` is width plus `2 / alpha` times the distance below the lower bound or above the upper bound. Use the same alpha as the interval construction. The legacy relative-width diagnostic divides by `abs(point_prediction) + 1e-10`; near-zero points therefore yield very large relative widths. Bound MSE and upper/point ratio are descriptive diagnostics, not coverage guarantees or proper point-prediction losses.

## Scope

This release implements unweighted split conformal absolute-residual regression and LAC/deterministic APS classification. It does not implement CV+, jackknife+, conformalized quantile regression, conditional/Mondrian calibration, online/time-series guarantees, weighted covariate-shift calibration, or randomized scores. Adding these needs separate calibration protocols and dedicated statistical validation.

Sklearn integration uses mixins before `BaseEstimator`, estimator cloning, validation, and fitted attributes. Point mode is the standard estimator contract; legacy conformal mode intentionally has a tuple prediction API and cannot be passed to ordinary prediction-based sklearn scorers. Wrap preprocessing inside the supplied estimator, particularly for automatic calibration.
