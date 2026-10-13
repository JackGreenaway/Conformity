# Diagnostics and errors

## Question: are calibrated regions useful?

Coverage alone can hide an unhelpful predictor that returns enormous regions. Evaluate on an untouched test set and pair coverage with efficiency:

| Diagnostic | Interpretation | Limitation |
| :--- | :--- | :--- |
| Interval coverage | Fraction of closed intervals containing their target | Finite test coverage fluctuates |
| Lower / upper miss rate | Fraction of targets below / above their interval | Tail comparisons are descriptive on a finite test set |
| Mean width | Average span in target units | Infinite bounds produce infinite width |
| Normalized mean width | Mean width divided by the test-target IQR | Infinite when the observed IQR is zero; report with point error and coverage-sensitive scores |
| Interval score | Width plus missed-bound distance penalty | Alpha must match interval construction; lower is better |
| Set coverage | Fraction containing the true label | Requires `classes` in membership-column order |
| Coverage by class | Set coverage among examples of each true class | Small or imbalanced classes produce noisy estimates; marginal calibration does not guarantee each class |
| Mean size | Average number of included classes | Small size can conceal undercoverage |
| Empty / singleton rate | Frequency of zero / one candidate | Neither gives conditional correctness |

```python
import numpy as np

from conformity import (
    interval_score,
    normalized_prediction_interval_width,
    prediction_interval_coverage,
    prediction_interval_lower_miss_rate,
    prediction_interval_upper_miss_rate,
    prediction_interval_width,
)
from conformity import (
    prediction_set_coverage,
    prediction_set_coverage_by_class,
    prediction_set_empty_rate,
    prediction_set_singleton_rate,
    prediction_set_size,
)

intervals = np.array([[0.0, 2.0], [1.0, 3.0]])
y = np.array([1.0, 4.0])
assert prediction_interval_coverage(y, intervals) == 0.5
assert prediction_interval_lower_miss_rate(y, intervals) == 0.0
assert prediction_interval_upper_miss_rate(y, intervals) == 0.5
assert prediction_interval_width(intervals) == 2.0
assert (
    normalized_prediction_interval_width([0, 0, 2, 2], [[0, 2], [0, 2], [0, 2], [0, 2]])
    == 1.0
)
assert interval_score(y, intervals, alpha=0.1) == 12.0
mask = np.array([[True, False], [False, False], [True, True]])
assert prediction_set_coverage(["a", "b", "b"], mask, classes=["a", "b"]) == 2 / 3
assert prediction_set_size(mask) == 1.0
assert prediction_set_empty_rate(mask) == 1 / 3
assert prediction_set_singleton_rate(mask) == 1 / 3
assert prediction_set_coverage_by_class(["a", "b", "b"], mask, classes=["a", "b"]) == {
    "a": 1.0,
    "b": 0.5,
}
```

## Results and next steps

The intervals cover one of two outcomes and have mean width 2. Their test-target IQR is 2 in the normalization example, so normalized mean width is 1. The interval example has no lower misses and one upper miss; that missed upper bound contributes a penalty of 20 for its row, giving a mean interval score of 12. The class example has one singleton, one empty set and one two-label set. Its class-wise coverage is 1.0 for `a` and 0.5 for `b`. These exact small examples verify conventions, not a statistical guarantee.

Use `evaluate` for a complete report. Regression includes R², MAE and MSE for point accuracy, plus coverage, tail miss rates, width, normalized width and interval score. Classification includes accuracy, log loss, overall coverage, set size, empty rate and singleton rate. A wide interval can still achieve high coverage: check point MAE/R² separately, and compare normalized width and interval score at the same alpha. Normalized width divides by the test-target IQR, so it is easier to compare across target units but still measures width alone; it is infinite when that IQR is zero. The interval score combines width with a penalty for missed outcomes. Use `prediction_set_coverage_by_class` to inspect empirical class-wise coverage and report each class's sample count with it. For asymmetric intervals, compare each tail miss rate with `alpha / 2`; symmetric intervals have one shared absolute-residual cutoff, so tail errors can be imbalanced. Examine subgroup coverage and distribution drift, while acknowledging that these summaries do not create subgroup guarantees. Recalibration needs fresh data independent of model fitting; it replaces previous scores and emits a warning.

## Common failures

| Symptom | Cause and action |
| --- | --- |
| `NotFittedError` | Fit before prediction; parameter changes invalidate fitting |
| Calibration `RuntimeError` | Call `calibrate` on independent data or enable automatic calibration |
| Infinite-threshold warning | Increase calibration size or choose a less extreme alpha |
| Unknown class label | Ensure calibration labels belong to fitted `classes_` |
| Invalid probabilities | Check shape, finite values, range and row sums in the wrapped classifier |
| Method changed after calibration | Refit/recalibrate with the intended score method |
| Invalid bounds or membership | Use ordered non-NaN `(n, 2)` bounds or a nonempty boolean mask |

See [API metric definitions](../api.md) and [calibration limits](calibration.md) before interpreting weighted or filtered evaluations.
