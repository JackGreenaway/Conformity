# Calibration and guarantees

## Question: what does a 90% interval promise?

Under exchangeability, an independently fitted score function gives marginal coverage at least $1-\alpha$ for a fresh observation. It does not give a 90% probability statement for an already observed interval or guarantee coverage for a selected subgroup.

With $n$ held-out scores, Conformity uses the rank

$$
k = \lceil (n+1)(1-\alpha) \rceil.
$$

If $k \le n$, the threshold is the $k$th sorted score; otherwise it is infinite. Score ties are included. Do not replace the corrected rank with a plain interpolated empirical quantile.

## Choosing a calibration size

For a wrapper on the final pipeline model, reserve calibration before fitting any learned preprocessing, set `auto_calibrate=False`, then calibrate on transformed held-out features. For automatic calibration with learned preprocessing, wrap the entire pipeline so its internal split precedes every fitting step. The [pipeline guide](estimators.md#why-the-split-matters) explains why these workflows differ.

A larger calibration partition gives a finer score quantile but leaves fewer fitting observations. `calibration_size` accepts a fraction or sample count. `random_state` controls the automatic split; `tts_kwargs` can override train/test split options. A reproducible split does not make temporal, grouped or shifted observations exchangeable.

For finite regression bounds you need at least $\alpha \ge 1/(n+1)$. Selecting alpha after viewing final calibration outcomes undermines the fixed-procedure interpretation. Repeated predictions can reuse stored scores without refitting, but their errors need not be independent.

## Limitations and next steps

Coverage is unweighted and marginal. Learned preprocessing, model tuning and method selection must stay independent of final calibration. Weighted metrics affect evaluation only. For drift, time series, covariate-conditioned guarantees or adaptive online calibration, this package's current methods are insufficient on their own.

Start with [the theory guide](../theory.md) for the intuition and coverage proof. Read the full [statistical specification](../legacy/THEORY.md), including its score definitions and proof. Evaluate coverage and efficiency together using [diagnostics](diagnostics.md).
