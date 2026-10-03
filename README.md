# Conformity

Split conformal prediction for sklearn-style regression and probabilistic classification estimators. Python 3.9+, NumPy 2+, and scikit-learn 1.6.1+.

```bash
pip install conformity-calib
```

## Regression

Use a sklearn pipeline with a point-mode conformal regressor, constructor-level automatic calibration, and explicit interval requests:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline as pipeline
from sklearn.preprocessing import StandardScaler
from conformity import ConformalRegressor

X, y = make_regression(n_samples=1000, noise=20, random_state=42)
X_new = X[:5]  # Replace with new observations in practice.

pipe = pipeline(
    StandardScaler(),
    ConformalRegressor(
        Ridge(),
        prediction_mode="point",
        auto_calibrate=True,
        random_state=42,
    ),
)

# During development: score point predictions.
results = cross_validate(pipe, X, y, cv=5, scoring="r2")

# Fit the final pipeline; calibration happens internally.
pipe.fit(X, y)
points = pipe.predict(X_new)
# Later: request intervals for your own testing.
points, intervals = pipe.predict(X_new, return_interval=True)
```

This is the default usage pattern in these docs. `pipeline` aliases sklearn’s `make_pipeline`, which accepts estimators directly and names steps automatically. Automatic calibration reserves 20% of each fit’s training observations by default. Prediction mode controls output independently of calibration. Point mode works with ordinary sklearn scorers; request intervals explicitly with `return_interval=True` (optionally pass `alpha`, which defaults to `0.05`). `cross_validate` fits clones, so fit the original pipeline before predicting.

The outer scaler fits on all observations passed to each pipeline fit, including the final wrapper's internal calibration features. This workflow is supported, but the usual split-conformal coverage proof does not apply to that placement of learned preprocessing. For that guarantee, put learned preprocessing inside the wrapped estimator, as described in [THEORY.md](THEORY.md#pipeline-placement-and-selection).

## Model-first configuration and sklearn pipelines

Pass a model, pipeline, or `GridSearchCV` as the estimator. Set `auto_calibrate=True` in the constructor so ordinary `fit(X, y)` splits, fits, and calibrates. The API default remains `False` for compatibility. Constructor `tts_kwargs` configures split options; explicit fit-time options override them.

With sklearn metadata routing disabled (the default), the outer pipeline forwards prediction keywords to the final wrapper. `return_interval=False` forces points; `True` returns points and intervals; omission follows `prediction_mode`. A standard pipeline does not forward custom `calibrate`, `predict_interval`, or `evaluate` methods: transform inputs with `pipe[:-1].transform(...)` before calling these methods on `pipe[-1]`.

For manual calibration, hold out calibration observations before fitting any pipeline steps, then call `pipe[-1].calibrate(pipe[:-1].transform(X_calib), y_calib)`. Calibration replaces previous scores and warns on replacement. Refitting or calling `set_params` invalidates calibration; `set_params` also invalidates the fitted model. A supplied fitted estimator is cloned and refitted; there is no prefit mode.

## Automatic calibration with `cross_validate`

Use `cross_validate(pipe, X, y, cv=5, scoring="r2")` as in the regression example. Each clone splits its fold's training observations into model-training and calibration partitions. Fold test observations are reserved for scoring. No separate `calibrate` call or fit-time calibration flag is needed.

Set `return_estimator=True` to inspect the fitted fold pipelines; each final wrapper has its own `n_calibration_` and calibration scores. A custom scorer receives `(fitted_pipeline, X_test, y_test)` and returns a scalar. For example:

```python
def coverage(pipe, X_test, y_test):
    return pipe[-1].evaluate(
        pipe[:-1].transform(X_test), y_test, alpha=0.1
    )["coverage"]

results = cross_validate(
    pipe, X, y, cv=5,
    scoring={"r2": "r2", "coverage": coverage},
    return_estimator=True,
    error_score="raise",
)
```

Cross-validation evaluates separate split-conformal models; it does not combine their scores into a CV+ predictor. Empirical coverage can differ from the nominal level. If selecting a model using these scores, reserve fresh calibration observations for its final coverage guarantee. For classification's marginal coverage interpretation, use fold splitting independent of outcomes, such as shuffled `KFold`; sklearn's default classifier folds are stratified and require care about sampling assumptions.

## Classification

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from conformity import ConformalClassifier

X, y = load_iris(return_X_y=True)
X_new = X[:5]  # Replace with new observations in practice.
pipe = pipeline(
    StandardScaler(),
    ConformalClassifier(
        LogisticRegression(max_iter=1000),
        method="aps",  # or "lac" (default)
        prediction_mode="point",
        auto_calibrate=True,
        random_state=42,
    ),
)
results = cross_validate(pipe, X, y, cv=5, scoring="accuracy")
pipe.fit(X, y)
labels = pipe.predict(X_new)
label_sets, probabilities = pipe.predict(X_new, return_set=True)
sets = pipe[-1].predict_set(pipe[:-1].transform(X_new))  # boolean (n, n_classes)
```

Labels can be strings or noncontiguous numbers. Set columns and probability columns follow `classes_`. Calibration labels absent from the fitted estimator are rejected. The estimator must implement `predict_proba` and expose `classes_`. For the coverage theorem, `classes_` must include every possible future label; unseen classes cannot be covered even by an all-class set. Probabilities must be finite, between zero and one, and sum to one per row.

* **LAC:** score a candidate class by `1 - probability`.
* **APS:** score it by cumulative descending probability up to and including that class. This implementation is deterministic and conservative; ties follow stable `classes_` order. It does not implement randomized APS or RAPS.

Empty sets are permitted. P-values count ties conservatively: `(1 + count(calibration_score >= candidate_score)) / (n + 1)`.

## sklearn compatibility and migration

The default `prediction_mode="conformal"` preserves the original API:

```python
legacy = ConformalRegressor(Ridge(), auto_calibrate=True, prediction_mode="conformal")
points, intervals = legacy.fit(X, y).predict(X_new, alpha=0.1)
```

For regression this returns `(points, intervals)`; for classification `(label_sets, probabilities)`. Legacy label sets use NaN for excluded classes, and object arrays for string labels. Prefer the boolean `predict_set` representation.

Use `prediction_mode="point"` for sklearn scorers, `GridSearchCV`, cross-validation, and ensembles. `predict_point` and `score` always use point predictions and do not require calibration. The explicit `predict_interval` / `predict_set` methods work in either mode and require calibration. In point mode, `alpha` has no effect on point predictions; it controls intervals or sets when explicitly requested.

```python
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(
    ConformalRegressor(Ridge(), prediction_mode="point"),
    {"estimator__alpha": [0.1, 1.0, 10.0]},
    scoring="neg_mean_absolute_error",
)
search.fit(X, y)
search.best_estimator_.calibrate(X_calib, y_calib)  # fresh, held-out observations
```

Both wrappers support cloning, nested estimator parameters, sparse CSR/CSC matrices, feature-count validation, and numeric DataFrames with feature-name validation. DataFrames are preserved for column-selecting pipelines. Sparse inputs still require support in the wrapped estimator. Multi-output regression, multilabel classification, precomputed kernels, metadata routing, and arbitrary structured/raw-text inputs are outside the current API.

## Controls and metrics

Constructor options are `estimator`, `auto_calibrate`, `tts_kwargs`, `calibration_size`, `random_state`, and `prediction_mode`; classifiers additionally expose `method`. `fit(..., auto_calibrate=True, tts_kwargs={...})` accepts `train_test_split` options overriding the constructor defaults. Splitting is random and unstratified by default. Stratifying calibration by outcome can change the exchangeability assumptions; use it deliberately rather than relying on an automatic classification default.

`fit(..., sample_weight=weights, **fit_params)` forwards estimator fitting options. Sample weights are validated and split alongside training observations. Other fit parameters, except step-qualified sample weights, are passed unchanged: provide metadata aligned to the training partition, or split manually. Pipelines require step-qualified parameters such as `ridge__sample_weight`; these weight vectors are validated and automatically sliced too. Pass them with their estimator-relative step names. The unqualified `sample_weight` is forwarded as supplied; it is not automatically routed to a pipeline step. Calibration is unweighted.

`evaluate(..., sample_weight=...)` returns:

| Task | Metrics |
| --- | --- |
| Regression | R², MAE, MSE, coverage, mean width, interval score |
| Classification | accuracy, log loss, coverage, mean set size, empty rate, singleton rate |

All conformal metrics are also available as standalone package exports and accept optional evaluation weights:

* `prediction_interval_coverage`, `prediction_interval_width`, `interval_score`
* `prediction_set_coverage`, `prediction_set_size`, `prediction_set_empty_rate`, `prediction_set_singleton_rate`
* Legacy diagnostics: `prediction_interval_efficiency`, `prediction_interval_ratio`, `prediction_interval_mse`, `prediction_set_efficiency`

Boolean set coverage requires `classes=pipe[-1].classes_`. Metrics accept lists and arrays, reject malformed/empty inputs, and validate weights. Weights change the reported empirical metric, not the conformal guarantee. Interval score combines width and missed-bound penalties; lower is better. Legacy set efficiency is `(size - 1)/(classes - 1)`, can be negative for empty sets, and now returns mean size for single-column sets instead of NaN. Interval ratio rejects zero point predictions rather than dividing by zero.

## Statistical contract

With `n` held-out calibration scores, the threshold is the `ceil((n + 1) * (1 - alpha))`-th order statistic, including an infinite sentinel at rank `n + 1`. There is no interpolation or clipping. If the requested rank exceeds `n`, regression intervals are unbounded and classification sets contain every fitted class; a warning explains why. Prediction includes candidates whose scores equal the threshold.

Under exchangeability of calibration and future observations conditional on the training information, with the score function fixed without calibration data, split conformal gives marginal coverage of at least `1 - alpha`. It does **not** promise coverage for every individual, subgroup, or realized test batch. Reusing training data as calibration, tuning on calibration outcomes, time dependence, and distribution shift can invalidate this guarantee. Keep an independent test set for evaluation. Extremely small alpha requires large calibration sets to produce finite thresholds.

These conventions follow the [split conformal regression literature](https://www.stat.berkeley.edu/~ryantibs/papers/conformal-jasa.pdf) and the [sklearn estimator interface](https://scikit-learn.org/stable/developers/develop.html). Sorted calibration scores are cached once; threshold lookup is constant-time, and p-values use binary search rather than sorting on every prediction.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for the API and [THEORY.md](THEORY.md) for score definitions, the rank proof, assumptions, pipeline placement, and limitations.

## Releases

Set `project.version` in `pyproject.toml`, commit the change, and push a matching `v`-prefixed tag:

```bash
git tag v0.1.2
git push origin v0.1.2
```

The release workflow validates the tag against the package version, runs lint and tests, builds and checks the wheel and source archive, publishes them to PyPI, and creates a GitHub release with generated notes and both distributions attached. Use a new version for each release; PyPI does not allow overwriting existing files.

Configure a `PYPI_API_TOKEN` repository or `pypi` environment secret, or configure [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/) for `.github/workflows/python-publish.yml` and the `pypi` environment and leave the secret unset. Any protection rules on the `pypi` environment apply before publishing.
