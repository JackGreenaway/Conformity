<div class="docs-hero" markdown="1">

# Conformity

<p class="lead">Turn a point prediction into an interval or a set of plausible labels.</p>

Conformity solves a practical gap in ordinary supervised learning: a model gives an answer, but a decision often needs a measure of uncertainty. It wraps a cloned scikit-learn estimator and uses independent held-out observations to calibrate prediction regions.

<div class="docs-actions" markdown="1">

[Get started](getting-started.md) [Choose a component](api.md)

</div>
</div>

## Why use it?

Choose a target miscoverage rate without specifying a noise distribution. Keep ordinary `predict` and `score` behavior for model development, then request intervals or class sets explicitly. Inspect coverage alongside width or set size on a separate test set.

| Question | Component | Next step |
| --- | --- | --- |
| How far could a numeric outcome be from the prediction? | `ConformalRegressor` | [Regression intervals](user-guide/regression.md) |
| Which class labels remain plausible? | `ConformalClassifier` | [Classification sets](user-guide/classification.md) |
| How do fitting and calibration interact? | `BaseConformalPredictor` | [Estimator lifecycle](user-guide/estimators.md) |
| Are uncertainty regions useful on unseen data? | Public metrics and `evaluate` | [Diagnostics](user-guide/diagnostics.md) |

## When the guarantee applies

Split conformal coverage is marginal over exchangeable calibration and future observations. It does not promise coverage for every individual, subgroup, or time period. Keep calibration observations out of model fitting, learned preprocessing, hyperparameter selection, and score-method selection. Distribution shift and dependent time-series data need a different justification.

Regression currently provides symmetric, constant-width, single-output intervals. Classification requires valid `predict_proba` output and uses LAC or deterministic APS; empty sets are valid outcomes. Tiny calibration sets can require infinite intervals or all-class sets.

Read [calibration and limitations](user-guide/calibration.md) before using the results to make decisions. The [original documentation](legacy/README.md), [API guide](legacy/DOCUMENTATION.md), and [statistical specification](legacy/THEORY.md) remain available in full.

## Understand the theory

Read [the theory behind conformal prediction](theory.md) for the intuition, finite-sample rank proof, regression and classification scores, and the limits of marginal coverage. A worked threshold calculation connects the mathematics to the returned prediction regions.
