# Metrics


Use an estimator's `evaluate` method for a ready-made report, or call a metric
function to build a custom report. These functions summarize predictions on
evaluation data; they do not provide a separate coverage guarantee. For set
metrics, the boolean columns must follow the supplied `classes` order.

```pycon
>>> from conformity import prediction_set_coverage, prediction_set_coverage_by_class
>>> y_true = ["billing", "account", "billing", "account"]
>>> sets = [[True, False], [False, True], [True, False], [True, False]]
>>> prediction_set_coverage(y_true, sets, classes=["billing", "account"])
0.75
>>> prediction_set_coverage_by_class(
...     y_true, sets, classes=["billing", "account"]
... )
{'billing': 1.0, 'account': 0.5}
```

| Metric family | Useful for | Reference |
| --- | --- | --- |
| Classification | Overall and per-class set coverage, set size, empty and singleton sets | [Classification metrics](#classification-metrics) |
| Regression | Interval coverage, lower/upper misses, width and interval score | [Regression metrics](#regression-metrics) |

<!-- BEGIN GENERATED REFERENCE -->
## Classification metrics

| Metric | Purpose |
| --- | --- |
| [`prediction_set_coverage`](metrics/prediction-set-coverage.md) | Fraction of sets containing the true class; supports arbitrary labels. |
| [`prediction_set_coverage_by_class`](metrics/prediction-set-coverage-by-class.md) | Return empirical set coverage for each represented class. |
| [`prediction_set_size`](metrics/prediction-set-size.md) | Mean cardinality (smaller is more efficient at comparable coverage). |
| [`prediction_set_empty_rate`](metrics/prediction-set-empty-rate.md) | Fraction of empty sets. |
| [`prediction_set_singleton_rate`](metrics/prediction-set-singleton-rate.md) | Fraction of sets containing exactly one label. |

## Regression metrics

| Metric | Purpose |
| --- | --- |
| [`prediction_interval_coverage`](metrics/prediction-interval-coverage.md) | Fraction of targets contained in closed prediction intervals. |
| [`prediction_interval_lower_miss_rate`](metrics/prediction-interval-lower-miss-rate.md) | Fraction of targets below their closed prediction intervals. |
| [`prediction_interval_upper_miss_rate`](metrics/prediction-interval-upper-miss-rate.md) | Fraction of targets above their closed prediction intervals. |
| [`prediction_interval_width`](metrics/prediction-interval-width.md) | Mean interval width; unbounded intervals have infinite width. |
| [`normalized_prediction_interval_width`](metrics/normalized-prediction-interval-width.md) | Mean interval width divided by the target's interquartile range. |
| [`interval_score`](metrics/interval-score.md) | Mean Winkler score: width plus 2/alpha times each missed-bound distance. |

<!-- END GENERATED REFERENCE -->
