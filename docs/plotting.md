# Plotting uncertainty

## Question

Are interval errors concentrated in a part of the feature space? A plot can reveal missed outcomes and show why marginal coverage does not imply local coverage. Matplotlib is optional; install it using [installation](installation.md).

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from conformity import ConformalRegressor

rng = np.random.default_rng(31)
X = rng.uniform(20, 500, size=(500, 1))  # Delivery distance in km.
y = 8 + 0.08 * X[:, 0] + rng.normal(0, 4, 500)  # Delivery time in hours.
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
# Automatic calibration holds out 20% of development rows before fitting.
model = ConformalRegressor(LinearRegression(), random_state=33).fit(X_dev, y_dev)
order = np.argsort(X_test[:, 0])
x = X_test[:, 0][order]
points, intervals = model.predict(X_test[order], alpha=0.1, return_interval=True)
observed = y_test[order]
fig, ax = plt.subplots()
ax.fill_between(x, intervals[:, 0], intervals[:, 1], alpha=0.25, label="90% interval")
ax.plot(x, points, label="Point prediction")
ax.scatter(x, observed, s=12, label="Observed test target")
ax.set(
    xlabel="Delivery distance (km)",
    ylabel="Delivery time (hours)",
    title="Held-out delivery uncertainty",
)
ax.legend()
assert len(ax.collections) == 2
fig.canvas.draw()
plt.close(fig)
```

## Configuration, results and interpretation

The simulated deliveries have distance in kilometres and outcomes in hours. `train_test_split` reserves 100 test deliveries; automatic calibration then reserves 80 of the 400 development rows and fits on 320. The seeds make the synthetic example reproducible. Sorting changes only display order; the model still receives complete test rows. The shaded region is a constant-width conformal interval, the line contains predictions, and dots contain unseen outcomes. The noninteractive backend permits execution in CI; in a notebook use its display backend and display `fig` before closing, or save with `fig.savefig("intervals.png")`.

## Next steps

Inspect misses by subgroup and compare coverage with width in [diagnostics](user-guide/diagnostics.md). A visual pattern is a prompt for investigation, not proof of conditional validity. For classification, plot set-size frequencies rather than assigning probabilities to membership masks.
