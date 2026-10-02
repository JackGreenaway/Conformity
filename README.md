# Conformity

Split conformal prediction for sklearn-style regression and probabilistic classification estimators. Python 3.9+, NumPy 2+, and scikit-learn 1.6.1+.

```bash
pip install conformity-calib
```

## Regression

Put preprocessing **inside** the wrapped pipeline. The automatic split then excludes calibration data from both preprocessing and model fitting.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from conformity import ConformalRegressor

X, y = make_regression(n_samples=1000, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
reg = ConformalRegressor(
    make_pipeline(StandardScaler(), Ridge()),
    calibration_size=0.25,
    random_state=42,
    prediction_mode="point",
)
reg.fit(X_train, y_train, auto_calibrate=True)
points = reg.predict(X_test)
bounds = reg.predict_interval(X_test, alpha=0.1)  # shape (n, 2)
report = reg.evaluate(X_test, y_test, alpha=0.1)
```

Alternatively, `fit(X_train, y_train)` followed by `calibrate(X_calib, y_calib)` gives full control over a separate calibration dataset. `calibrate` replaces previous scores and warns on replacement. Refitting or calling `set_params` invalidates calibration; `set_params` also invalidates the fitted model.

## Model-first configuration and sklearn pipelines

Pass your model, pipeline, or `GridSearchCV` as the estimator. Set `auto_calibrate=True` in the constructor to make ordinary `fit(X, y)` split, fit, and calibrate. The default remains `False` for compatibility. Constructor `tts_kwargs` configures split options; explicit fit-time options override them.

```python
reg = ConformalRegressor(
    make_pipeline(StandardScaler(), Ridge()),
    auto_calibrate=True,
    calibration_size=0.25,
    random_state=42,
    prediction_mode="point",
)
reg.fit(X_train, y_train)
points = reg.predict(X_test)
bounds = reg.predict_interval(X_test, alpha=0.1)
```

Wrapping a `GridSearchCV` runs its tuning only on the training partition, keeping the internal calibration partition out of preprocessing and tuning. This performs split conformal calibration, not cross-conformal prediction.

Either wrapper can also be the final step of a standard sklearn pipeline:

```python
pipe = make_pipeline(
    StandardScaler(),
    ConformalRegressor(Ridge(), prediction_mode="point"),
)
pipe.fit(X_train, y_train)
# X_calib must be held out before fitting any pipeline steps.
pipe[-1].calibrate(pipe[:-1].transform(X_calib), y_calib)
points = pipe.predict(X_test)
points, bounds = pipe.predict(X_test, return_interval=True, alpha=0.1)
```

With sklearn metadata routing disabled (the default), the outer pipeline forwards prediction keywords to the regressor. `return_interval=False` forces point predictions; `True` returns points and intervals; omission preserves `prediction_mode`. A standard pipeline does not forward custom `calibrate`, `predict_interval`, or `evaluate` methods: transform inputs using its fitted preprocessing before calling these on the final estimator. For internal splitting, wrap the complete preprocessing pipeline as in the first example; splitting only inside the final step lets preceding transformers fit on the calibration observations.

Classification supports the same outer-pipeline workflow:

```python
from sklearn.linear_model import LogisticRegression
from conformity import ConformalClassifier

pipe = make_pipeline(
    StandardScaler(),
    ConformalClassifier(LogisticRegression(max_iter=1000), prediction_mode="point"),
)
pipe.fit(X_train, y_train)  # classification data; calibration held out first
pipe[-1].calibrate(pipe[:-1].transform(X_calib), y_calib)
labels = pipe.predict(X_test)
label_sets, probabilities = pipe.predict(X_test, return_set=True, alpha=0.1)
mask = pipe[-1].predict_set(pipe[:-1].transform(X_test), alpha=0.1)
```

`return_set=False` forces point labels; omission preserves `prediction_mode`. The wrapper always requires an estimator, including when it is a pipeline step. A supplied fitted model is cloned and refitted; there is no prefit mode.

## Automatic calibration with `cross_validate`

Set `auto_calibrate=True` on the wrapper and `prediction_mode="point"` for ordinary sklearn scorers. Each cross-validation clone splits its fold's training observations into model-training and calibration partitions, fits the estimator, then calibrates automatically. The fold's test observations are reserved for scoring. No separate `calibrate` call or fit-time calibration flag is needed. This follows sklearn's [`cross_validate` estimator and scorer interface](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).

The wrapper can be the last step of a standard `Pipeline`. Put learned preprocessing inside its estimator so automatic splitting happens before that preprocessing is fitted:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from conformity import ConformalRegressor

X, y = make_regression(n_samples=600, noise=20, random_state=42)
pipe = Pipeline([
    ("conformal", ConformalRegressor(
        estimator=make_pipeline(StandardScaler(), Ridge()),
        auto_calibrate=True,
        calibration_size=0.2,
        random_state=42,
        prediction_mode="point",
    )),
])

# Custom scorers receive the fitted pipeline and that fold's test data.
def coverage(estimator, X_test, y_test):
    return estimator.named_steps["conformal"].evaluate(
        X_test, y_test, alpha=0.1
    )["coverage"]

results = cross_validate(
    pipe,
    X,
    y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring={"mae": "neg_mean_absolute_error", "coverage": coverage},
    return_estimator=True,
    error_score="raise",
)
print(-results["test_mae"])
print(results["test_coverage"])

# Each returned pipeline contains its own fitted, calibrated wrapper.
fold_model = results["estimator"][0].named_steps["conformal"]
print(fold_model.n_calibration_)

# cross_validate fits clones; fit the original pipeline for later use.
pipe.fit(X, y)
points, intervals = pipe.predict(X[:5], return_interval=True, alpha=0.1)
```

For classification, use `ConformalClassifier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), auto_calibrate=True, prediction_mode="point", random_state=42)` as the final step and use an ordinary scorer such as `"accuracy"` or `"neg_log_loss"`. The coverage scorer above also works for classification. Class sets are available through `pipe.predict(X_new, return_set=True, alpha=0.1)`. For the marginal coverage interpretation, use fold splitting independent of outcomes, such as shuffled `KFold`; sklearn's default classifier folds are stratified and require care about the sampling assumptions.

`Pipeline([("conformal", ConformalRegressor(Ridge(), auto_calibrate=True, prediction_mode="point"))])` is sufficient when there is no learned preprocessing. An outer pipeline with `StandardScaler()` before `ConformalRegressor(Ridge(), auto_calibrate=True)` runs in `cross_validate`, but the scaler sees the internal calibration features before splitting, so the usual split-conformal coverage proof does not apply. The final wrapper cannot control fitting of earlier outer steps. Use the nested estimator shown above for learned transformations; preceding stateless transformations are also safe.

Cross-validation evaluates separate split-conformal models; it does not combine their calibration scores into a CV+ predictor. Each fold's empirical test coverage can differ from the nominal level. If you choose a model using these scores, reserve fresh calibration observations for the selected model's final coverage guarantee.

## Classification

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from conformity import ConformalClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = ConformalClassifier(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    method="aps",  # or "lac" (default)
    random_state=42,
    prediction_mode="point",
)
clf.fit(X_train, y_train, auto_calibrate=True)
labels = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
sets = clf.predict_set(X_test, alpha=0.1)  # boolean (n, n_classes)
label_sets = [clf.classes_[row].tolist() for row in sets]
p_values = clf.predict_p_values(X_test)
report = clf.evaluate(X_test, y_test, alpha=0.1)
```

Labels can be strings or noncontiguous numbers. Set columns and probability columns follow `classes_`. Calibration labels absent from the fitted estimator are rejected. The estimator must implement `predict_proba` and expose `classes_`. For the coverage theorem, `classes_` must include every possible future label; unseen classes cannot be covered even by an all-class set. Probabilities must be finite, between zero and one, and sum to one per row.

* **LAC:** score a candidate class by `1 - probability`.
* **APS:** score it by cumulative descending probability up to and including that class. This implementation is deterministic and conservative; ties follow stable `classes_` order. It does not implement randomized APS or RAPS.

Empty sets are permitted. P-values count ties conservatively: `(1 + count(calibration_score >= candidate_score)) / (n + 1)`.

## sklearn compatibility and migration

The default `prediction_mode="conformal"` preserves the original API:

```python
points, intervals = reg.set_params(prediction_mode="conformal").fit(
    X_train, y_train, auto_calibrate=True
).predict(X_test, alpha=0.1)
```

For regression this returns `(points, intervals)`; for classification `(label_sets, probabilities)`. Legacy label sets use NaN for excluded classes, and object arrays for string labels. Prefer the boolean `predict_set` representation.

Use `prediction_mode="point"` for sklearn scorers, `GridSearchCV`, cross-validation, and ensembles. `predict_point` and `score` always use point predictions and do not require calibration. The explicit `predict_interval` / `predict_set` methods work in either mode and require calibration. In point mode, the `alpha` argument to `predict` is ignored; pass it to the explicit conformal methods.

```python
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(
    ConformalRegressor(Ridge(), prediction_mode="point"),
    {"estimator__alpha": [0.1, 1.0, 10.0]},
    scoring="neg_mean_absolute_error",
)
search.fit(X_train, y_train)
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

Boolean set coverage requires `classes=clf.classes_`. Metrics accept lists and arrays, reject malformed/empty inputs, and validate weights. Weights change the reported empirical metric, not the conformal guarantee. Interval score combines width and missed-bound penalties; lower is better. Legacy set efficiency is `(size - 1)/(classes - 1)`, can be negative for empty sets, and now returns mean size for single-column sets instead of NaN. Interval ratio rejects zero point predictions rather than dividing by zero.

## Statistical contract

With `n` held-out calibration scores, the threshold is the `ceil((n + 1) * (1 - alpha))`-th order statistic, including an infinite sentinel at rank `n + 1`. There is no interpolation or clipping. If the requested rank exceeds `n`, regression intervals are unbounded and classification sets contain every fitted class; a warning explains why. Prediction includes candidates whose scores equal the threshold.

Under exchangeability of calibration and future observations conditional on the training information, with the score function fixed without calibration data, split conformal gives marginal coverage of at least `1 - alpha`. It does **not** promise coverage for every individual, subgroup, or realized test batch. Reusing training data as calibration, tuning on calibration outcomes, time dependence, and distribution shift can invalidate this guarantee. Keep an independent test set for evaluation. Extremely small alpha requires large calibration sets to produce finite thresholds.

These conventions follow the [split conformal regression literature](https://www.stat.berkeley.edu/~ryantibs/papers/conformal-jasa.pdf) and the [sklearn estimator interface](https://scikit-learn.org/stable/developers/develop.html). Sorted calibration scores are cached once; threshold lookup is constant-time, and p-values use binary search rather than sorting on every prediction.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests/test_safety_and_api.py
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
