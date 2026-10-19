# Regression intervals

## When to use ConformalRegressor

Use it for one numeric target. The wrapped estimator must implement `fit` and return one finite value per sample from `predict`. The default `interval_method="symmetric"` uses one residual quantile for equal lower and upper corrections. Set `interval_method="asymmetric"` to calibrate the two sides separately.

With the default symmetric method, calibration stores absolute residuals. For miscoverage $\alpha$, the threshold $q$ is the finite-sample order statistic and a prediction region is $[\hat y-q, \hat y+q]$.

The asymmetric method stores signed residuals. It selects a lower and upper correction using a one-sided conformal quantile at miscoverage $\alpha/2$ for each tail, then returns $[\hat y-q_{\mathrm{lower}},\ \hat y+q_{\mathrm{upper}}]$. A union bound gives marginal coverage of at least $1-\alpha$ under exchangeability. The tail allocation is fixed at equal halves; this method does not estimate conditional coverage or adapt width to each input.

| Task | Method | Result |
| --- | --- | --- |
| Get ordinary predictions | `predict(X)` or `predict_point(X)` | `(n_samples,)` points |
| Get bounds | `predict_interval(X, alpha)` | `(n_samples, 2)` lower/upper bounds |
| Get both | `predict(X, alpha, return_interval=True)` | `(points, intervals)` |
| Measure uncertainty quality | `evaluate(X, y, alpha)` | R², MAE, MSE, coverage, lower/upper miss rates, mean and normalized width, interval score |

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalRegressor

X, y = make_regression(n_samples=500, n_features=4, noise=10, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=13
)
model = ConformalRegressor(
    Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
    interval_method="asymmetric",
    tts_kwargs={"random_state": 0},
)
model.fit(X_train, y_train)  # automatic split happens before pipeline fitting
points, intervals = model.predict(X_test, alpha=0.1, return_interval=True)
assert intervals.shape == (len(X_test), 2)
```

For the asymmetric method, each calibration tail uses the corrected rank $\lceil(n+1)(1-\alpha/2)\rceil$. A finite correction therefore requires $\alpha \geq 2/(n+1)$; otherwise the corresponding bound is infinite.

## Limitations and choices

Both methods produce fixed corrections across rows at a fixed alpha; only the asymmetric method allows the lower and upper correction sizes to differ. Neither is quantile regression or supports multi-output targets, and neither guarantees conditional coverage in heteroscedastic data. Too few calibration samples yield infinite bounds rather than a falsely narrow interval.

| Method | Advantages | Costs and limits |
| --- | --- | --- |
| Symmetric | One absolute-residual cutoff; equal-width corrections; finite bounds need $\alpha \geq 1/(n+1)$. | Cannot allocate more width to one tail; a large residual on either side widens both sides. |
| Asymmetric | Separate lower and upper corrections can reflect directional residual imbalance and may improve width when that imbalance matters. | No promise of shorter width; tail corrections use the more extreme $\alpha/2$ rank and require $\alpha \geq 2/(n+1)$ for finite bounds; still constant across inputs. |

Treat the choice as a modeling decision, not a guarantee of efficiency. Compare both at the same alpha on data kept separate from fitting and calibration. See the [theory guide](../theory.md#regression-absolute-residuals) for the score definitions and comparison plot.

Ordinary `score` is R² and uses point predictions; it does not evaluate interval quality. Match the interval score's alpha to the alpha used for prediction. Compare models at the same alpha using coverage, width and interval score. For asymmetric intervals, inspect lower and upper miss rates separately against the nominal `alpha / 2` per tail; on finite test samples these rates vary, and for symmetric intervals the two tails are not controlled separately.

## Next steps

Run [regression decisions](../examples/regression.md), [plot the results](../plotting.md), and inspect the [ConformalRegressor reference](../classes/conformal-regressor.md).
