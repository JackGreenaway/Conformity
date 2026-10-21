# User guide

Conformity separates fitting a point model from calibrating its uncertainty.
Start with the lifecycle, choose a region suited to your target, then evaluate
coverage and usefulness on observations used by neither step.

1. [Estimators and pipelines](estimators.md): cloning, fitting, calibration,
   preprocessing placement and model selection.
2. [Calibration and guarantees](calibration.md): held-out splits, exchangeability,
   finite-sample thresholds and the limits of marginal coverage.
3. [Regression intervals](regression.md): symmetric or asymmetric residual
   corrections, marginal coverage and the tradeoff between coverage and width.
4. [Classification sets](classification.md): LAC and deterministic APS,
   class ordering, p-values and ambiguous predictions.
5. [Diagnostics and errors](diagnostics.md): coverage, efficiency, weights,
   failure modes and independent evaluation.
6. [Theory](../theory.md): score definitions, order statistics and the rank proof.
7. [Plotting](../plotting.md): visualize intervals and investigate missed outcomes.

For complete workflows, visit the [examples gallery](../examples/index.md).
The [API reference](../api/index.md) describes callable contracts and fitted state.
