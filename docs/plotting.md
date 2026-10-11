# Plotting uncertainty

## Question

Are interval errors concentrated in a part of the feature space? A plot can reveal missed outcomes and show why marginal coverage does not imply local coverage. Matplotlib is optional; install it using [installation](installation.md).

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from conformity import ConformalRegressor

rng = np.random.default_rng(31)
X = rng.uniform(-3, 3, size=(500, 1))
y = 2 * X[:, 0] + rng.normal(size=500)
model = ConformalRegressor(LinearRegression(), random_state=32).fit(X[:400], y[:400])
order = np.argsort(X[400:, 0])
x = X[400:, 0][order]
points, intervals = model.predict(X[400:][order], alpha=0.1, return_interval=True)
observed = y[400:][order]
fig, ax = plt.subplots()
ax.fill_between(x, intervals[:, 0], intervals[:, 1], alpha=0.25, label="90% interval")
ax.plot(x, points, label="Point prediction")
ax.scatter(x, observed, s=12, label="Observed test target")
ax.set(xlabel="Feature", ylabel="Target", title="Held-out regression uncertainty")
ax.legend()
assert len(ax.collections) == 2
fig.canvas.draw()
plt.close(fig)
```

## Configuration, results and interpretation

The seed makes the synthetic example reproducible. Sorting changes only display order; the model still receives complete test rows. The shaded region is a constant-width conformal interval, the line contains predictions, and dots contain unseen outcomes. The noninteractive backend permits execution in CI; in a notebook use its display backend and display `fig` before closing, or save with `fig.savefig("intervals.png")`.

## Next steps

Inspect misses by subgroup and compare coverage with width in [diagnostics](user-guide/diagnostics.md). A visual pattern is a prompt for investigation, not proof of conditional validity. For classification, plot set-size frequencies rather than assigning probabilities to membership masks.
