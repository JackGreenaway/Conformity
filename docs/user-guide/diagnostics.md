# Diagnostics and errors

## Question: are calibrated regions useful?

Coverage alone can hide an unhelpful predictor that returns enormous regions. Evaluate on an untouched test set and pair coverage with efficiency:

| Diagnostic | Interpretation | Limitation |
| --- | --- | --- |
| Interval coverage | Fraction of closed intervals containing their target | Finite test coverage fluctuates |
| Mean width | Average span in target units | Infinite bounds produce infinite width |
| Interval score | Width plus missed-bound distance penalty | Alpha must match interval construction; lower is better |
| Set coverage | Fraction containing the true label | Requires `classes` in membership-column order |
| Mean size | Average number of included classes | Small size can conceal undercoverage |
| Empty / singleton rate | Frequency of zero / one candidate | Neither gives conditional correctness |

```python
import numpy as np

from conformity import (
    interval_score,
    prediction_interval_coverage,
    prediction_interval_width,
)
from conformity import (
    prediction_set_coverage,
    prediction_set_empty_rate,
    prediction_set_singleton_rate,
    prediction_set_size,
)

intervals = np.array([[0.0, 2.0], [1.0, 3.0]])
y = np.array([1.0, 4.0])
assert prediction_interval_coverage(y, intervals) == 0.5
assert prediction_interval_width(intervals) == 2.0
assert interval_score(y, intervals, alpha=0.1) == 12.0
mask = np.array([[True, False], [False, False], [True, True]])
assert prediction_set_coverage(["a", "b", "b"], mask, classes=["a", "b"]) == 2 / 3
assert prediction_set_size(mask) == 1.0
assert prediction_set_empty_rate(mask) == 1 / 3
assert prediction_set_singleton_rate(mask) == 1 / 3
```

## Results and next steps

The intervals cover one of two outcomes and have mean width 2. The missed upper bound contributes a penalty of 20 for its row, giving a mean interval score of 12. The class example has one singleton, one empty set and one two-label set. These exact small examples verify conventions, not a statistical guarantee.

Use `evaluate` for a complete report. Examine subgroup coverage and distribution drift, while acknowledging that the package gives no subgroup guarantee. Recalibration needs fresh data independent of model fitting; it replaces previous scores and emits a warning.

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
