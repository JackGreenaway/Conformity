# Statistical specification

## Default usage and coverage assumptions

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

## Estimator and data contract

Both wrappers require an estimator and fit a fresh sklearn clone. A supplied fitted estimator is therefore refitted, not used in a prefit mode. Regression requires finite single-output predictions; classification requires probabilities and `classes_`. The complete learned score function, including preprocessing, feature selection, probability calibration and hyperparameter tuning, must be fixed without using the conformal calibration observations.

For the theorem below, conditional on the training information, calibration observations and the future observation must be exchangeable. Independent, identically distributed observations with a split selected independently of their values suffice. General exchangeable data can also satisfy this condition; independence between calibration observations is not required. Training weights may change the fitted function but do not produce weighted conformal calibration.

Classification additionally requires that `classes_` contains the entire possible future label universe. A rare class missing from training cannot be returned by this implementation, even with an infinite threshold. Rejecting unknown labels at calibration detects some failures of this requirement but cannot establish it. Do not remove unknown-label calibration rows to make calibration succeed.

## Scores and prediction regions

Let the fitted model be fixed and let `S(x, y)` be a nonconformity score, with larger values meaning worse agreement. The implemented scores are:

| Method | Score | Returned region |
| --- | --- | --- |
| Absolute-residual regression | `abs(y - f(x))` | `[f(x) - q, f(x) + q]` |
| LAC classification | `1 - p_y(x)` | Labels with `1 - p_y(x) <= q` |
| Deterministic APS | Sum of probabilities through label `y` in descending order | Labels whose cumulative score is `<= q` |

APS uses stable `classes_` order to break equal-probability ties, identically at calibration and prediction. This is a deterministic cumulative-score variant; it does not implement the randomized boundary rule in Romano, Sesia and Candès. It can produce empty sets. It need not include the first label crossing `q`. Probabilities need not be statistically calibrated for the rank guarantee; model quality affects set efficiency. Regression intervals have constant width at a given alpha and do not adapt to heteroscedasticity.

## Finite-sample rank argument

For `n` calibration scores, write their sorted values as `s_(1), ..., s_(n)`. For a fixed `0 < alpha < 1`, define

```text
k = ceil((n + 1) * (1 - alpha))
q = s_(k) when k <= n, otherwise +infinity
C(x) = {y : S(x, y) <= q}
```

Under the contract above, the `n + 1` scores, including the future true-label score, are exchangeable. Break score ties with independent random ranks for the proof only. The future rank is uniform on `1, ..., n + 1`; every rank at most `k` is covered by the inclusive score rule. Hence

```text
P(Y_new in C(X_new)) >= k / (n + 1) >= 1 - alpha.
```

If scores are almost surely distinct, coverage equals `k / (n + 1)`; ties can make coverage larger, including one. Thus the guarantee is a lower bound, not exact nominal coverage. With `k = n + 1`, the full response space is required: all real numbers for regression and all labels in the assumed classification universe. A finite threshold requires `alpha >= 1 / (n + 1)` in exact arithmetic. The implementation uses NumPy floating-point arithmetic for the rank calculation.

This probability averages over calibration and future observations, conditional on training when the conditional exchangeability assumption holds. It is not a guarantee conditional on the realized calibration sample, a particular feature value, subgroup, or entire test batch. Each fixed alpha has the guarantee; choosing alpha or the score method after inspecting calibration outcomes needs additional theory. Reusing a fixed calibration set for multiple predictions provides marginal guarantees, not simultaneous coverage of every prediction.

## P-values

Classification returns conservative p-values

```text
p(x, y) = (1 + number of calibration scores >= S(x, y)) / (n + 1).
```

For the true label these are super-uniform under the same assumptions: `P(p(X_new, Y_new) <= alpha) <= alpha`. Counting ties with `>=` is essential. The score-threshold region agrees with `p(x, y) > alpha` in exact arithmetic. These are conformal plausibility values, not posterior label probabilities.

## Pipeline placement and selection

For the coverage theorem, use `ConformalRegressor(Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]), prediction_mode="point", auto_calibrate=True, random_state=42)` and its classification equivalent. This is the alternative to the default outer-preprocessing workflow when calibration must be excluded from every learned step. Internal splitting then happens before every fitted pipeline step. Wrapping `GridSearchCV(Pipeline(...))` also keeps calibration out of tuning. Tuning a wrapper externally and then reusing its internal calibration split can introduce selection dependence; recalibrate on fresh held-out data.

An outer `Pipeline([("preprocessing", preprocessing), ("conformal", wrapper)])` is valid with manual calibration held out before `Pipeline.fit`: transform calibration features using the fitted prefix, then call the final wrapper's `calibrate`. Automatically splitting only inside the final wrapper allows preceding learned transformers to see calibration features. The standard proof above then does not apply. Stateless transformations are safe. Outcome stratification, ordered splitting and grouped sampling need assumptions appropriate to their sampling scheme; the default random, unstratified split avoids those extra requirements for iid data.

## Evaluation and limits

Empirical coverage is a sample diagnostic and may fall below nominal coverage in a finite test batch. Evaluation weights change the diagnostic target, not the unweighted coverage theorem. Compare widths or set sizes at comparable coverage. The interval score is width plus `2 / alpha` times each missed-bound distance; use the construction alpha. It is a proper score for equal-tailed quantile intervals, but residual conformal intervals need not estimate those quantiles.

No distribution-shift, time-series, subgroup-conditional, weighted conformal, CV+, jackknife+, CQR, or randomized APS guarantee is implemented.

## Primary references

- [Lei et al. (2018), Distribution-Free Predictive Inference for Regression](https://arxiv.org/abs/1604.04173): split-conformal residual intervals and finite-sample coverage.
- [Romano, Sesia and Candès (2020), Classification with Valid and Adaptive Coverage](https://papers.neurips.cc/paper_files/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html): adaptive classification scores and the randomized procedure from which the deterministic variant differs.
- [scikit-learn Pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html): step-qualified fitting parameters and prediction forwarding.
