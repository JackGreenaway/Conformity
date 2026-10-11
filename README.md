<div align="center">

# Conformity

**Point predictions with calibrated uncertainty.**

Regression intervals · Classification sets · Scikit-learn workflows

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://github.com/JackGreenaway/Conformity/actions/workflows/tests.yml/badge.svg)](https://github.com/JackGreenaway/Conformity/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/conformity-calib?style=flat-square)](https://pypi.org/project/conformity-calib/)

[Quick start](#quick-start) · [Documentation](https://jackgreenaway.github.io/Conformity/) · [API reference](docs/api.md) · [Theory](THEORY.md) · [Development](docs/development.md)

</div>

---

A point model tells you its best answer. Conformity helps you ask **how uncertain that answer is**, using independent held-out observations to calibrate prediction intervals or sets of plausible class labels. Wrap a scikit-learn estimator or pipeline, keep ordinary point predictions for model development, and request conformal output when your decision needs a range or several candidates.

## Installation

```bash
uv add conformity-calib
# Or:
pip install conformity-calib
```

Requires Python **3.9+**, NumPy >=2.0.2 and scikit-learn >=1.6.1. Install `matplotlib` separately for [plotting](docs/plotting.md).

<a id="regression"></a>
<a id="classification"></a>
<a id="model-first-configuration-and-sklearn-pipelines"></a>
<a id="automatic-calibration-with-cross_validate"></a>
<a id="sklearn-integration"></a>
<a id="controls-and-metrics"></a>

## Choose a component

| Question | Component | Guide |
| --- | --- | --- |
| What range could contain a numeric outcome? | `ConformalRegressor` | [Regression](docs/user-guide/regression.md) |
| Which class labels remain plausible? | `ConformalClassifier` | [Classification](docs/user-guide/classification.md) |
| How are fitting and calibration managed? | `BaseConformalPredictor` | [Estimator lifecycle](docs/user-guide/estimators.md) |
| Are regions useful on unseen data? | `evaluate` and public metrics | [Diagnostics](docs/user-guide/diagnostics.md) |

## Quick start

Reserve a test set for evaluation. Put learned preprocessing **inside the wrapped estimator** so it never fits on calibration observations. Here `alpha=0.1` requests 90% marginal coverage:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalRegressor

X, y = make_regression(n_samples=1000, n_features=5, noise=15, random_state=7)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
model = ConformalRegressor(
    make_pipeline(StandardScaler(), Ridge()),
    calibration_size=0.25,
    random_state=9,
).fit(X_dev, y_dev)

points, intervals = model.predict(X_test, alpha=0.1, return_interval=True)
assert points.shape == (250,)
assert intervals.shape == (250, 2)
print(model.evaluate(X_test, y_test, alpha=0.1))
```

The report pairs point error with coverage, width and interval score. Bounds are symmetric and constant-width at a fixed alpha; lower miscoverage usually makes them wider. Test coverage fluctuates around the population target. The [complete walkthrough](docs/getting-started.md) explains the configuration, results and next steps.

For classification, `predict_set` returns a boolean membership matrix in fitted `classes_` order. `predict(..., return_set=True)` returns **(membership, probabilities)**. Run the [classification example](docs/examples/classification.md) to interpret label sets, empty sets and review decisions.

<a id="statistical-contract"></a>

## Guarantees and limitations

Coverage is **marginal under exchangeability**, with fitting and model selection independent of final calibration. It does not guarantee coverage for each individual or subgroup. Distribution shift, time dependence and calibration leakage can invalidate the guarantee. Very small calibration sets can require infinite intervals or all-class sets.

Regression supports one numeric target with symmetric absolute-residual intervals. Classification supports LAC and deterministic APS scores and requires valid `predict_proba` output. Calibration is unweighted; weights affect fitting or evaluation only.

Read [calibration and assumptions](docs/user-guide/calibration.md) and the [statistical theory](THEORY.md) before making decisions from uncertainty output.

## Documentation

The [documentation site](https://jackgreenaway.github.io/Conformity/) covers installation, plotting, guides, estimator classes, diagnostics, runnable examples, generated API reference and development. It becomes available after the first successful Pages deployment.

Start with [getting started](docs/getting-started.md), then [examples](docs/examples/index.md). The [original README workflows](docs/legacy/README.md), [existing API guide](DOCUMENTATION.md), and [theory](THEORY.md) are preserved in full.

<a id="development"></a>
<a id="releases"></a>

## Development and releases

```bash
uv python install 3.12
uv sync --locked --group docs --group release --python 3.12
bash scripts/validate.sh
```

CI tests Python 3.9, 3.10, 3.11 and 3.12. Documentation and release tooling use locked dependencies on Python 3.12. See [development and deployment](docs/development.md) for local preview, reference generation, distribution builds, GitHub Pages settings and PyPI trusted publishing.

For a release, update `project.version` and `conformity.__version__` together, refresh `uv.lock`, validate and push the revision, then run **Create GitHub Release**. It rejects existing tags and publishes checked artifacts through the separate PyPI workflow. No release is created by the local build commands above.
