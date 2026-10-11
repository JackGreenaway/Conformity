# Regression intervals

## When to use ConformalRegressor

Use it for one numeric target when symmetric residual intervals are an acceptable summary. The wrapped estimator must implement `fit` and return one finite value per sample from `predict`.

Calibration stores absolute residuals. For miscoverage $\alpha$, the threshold $q$ is the finite-sample order statistic and a prediction region is $[\hat y-q, \hat y+q]$.

| Task | Method | Result |
| --- | --- | --- |
| Get ordinary predictions | `predict(X)` or `predict_point(X)` | `(n_samples,)` points |
| Get bounds | `predict_interval(X, alpha)` | `(n_samples, 2)` lower/upper bounds |
| Get both | `predict(X, alpha, return_interval=True)` | `(points, intervals)` |
| Measure uncertainty quality | `evaluate(X, y, alpha)` | R², MAE, MSE, coverage, width, interval score |

## Limitations and choices

The interval width is constant across rows at a fixed alpha. This is not a quantile regression method, does not support multi-output targets, and does not guarantee conditional coverage in heteroscedastic data. Very small alpha with too few calibration samples yields infinite bounds rather than a falsely narrow interval.

Ordinary `score` is R² and uses point predictions; it does not evaluate interval quality. Match the interval score's alpha to the alpha used for prediction. Compare models at the same alpha using both coverage and width.

## Next steps

Run [regression decisions](../examples/regression.md), [plot the results](../plotting.md), and inspect the [ConformalRegressor reference](../classes/conformal-regressor.md).
