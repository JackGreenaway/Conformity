# Classification sets

## When to use ConformalClassifier

Use it when a categorical decision should expose more than one plausible answer. The estimator must provide finite probabilities in `[0, 1]`, one column per fitted class, with each row summing to one.

| Choice | Score | When to use it | Limitation |
| --- | --- | --- | --- |
| `method="lac"` | One minus candidate probability | A simple probability cutoff calibrated on held-out labels | Can produce empty sets and ignores probability ranking structure |
| `method="aps"` | Descending cumulative probability through the candidate | When ranked probability mass is a useful score | Deterministic variant, stable ties in class order; differs from randomized APS |

Sets include candidate scores at or below the threshold. `predict_set` returns a boolean `(n_samples, n_classes)` membership matrix; always map its columns through `classes_`. `predict_proba` returns probabilities, `predict` returns ordinary labels, and `predict(..., return_set=True)` returns **(mask, probabilities)**. Calibration labels must belong to fitted classes.

`predict_p_values` returns conformal tail-rank p-values for every candidate, including a finite-sample correction and inclusive ties. They are not posterior class probabilities. Changing the score method requires recalibration; `set_params` also removes fitted state and requires refitting.

For an imbalanced test set, inspect `prediction_set_coverage_by_class` alongside overall coverage and each class's sample count. These are empirical subgroup rates: marginal split calibration does not promise the target coverage within every class. See the [diagnostics guide](diagnostics.md) for an example.

## Rare classes and automatic calibration

Automatic calibration defaults to a random 80/20 split. For classification, `warn_on_rare_classes=True` checks that every observed class appears in both the training and calibration partitions. If a class is missing from either side, it warns and suggests a stratified splitter. This warning is enabled by default and can be disabled with `warn_on_rare_classes=False`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from conformity import ConformalClassifier

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
model = ConformalClassifier(
    LogisticRegression(),
    calibration_splitter=splitter,
    warn_on_rare_classes=False,
)
```

Use `StratifiedShuffleSplit` when preserving class representation is important for fitting and calibration. Stratification cannot place a class in both partitions when there are too few examples, and it does not create class-conditional coverage. It is a label-aware engineering choice; this implementation's standard finite-sample guarantee is documented for the default random split under exchangeability, not for label-stratified splitting. For formal class-wise coverage, use a method that calibrates each class separately.

## Limitations

A singleton is not an individual correctness guarantee. Empty sets and all-class sets are permitted. Tiny calibration sets can require including every class. Tune the point model and choose LAC or APS without looking at the final calibration or test outcomes.

## Next steps

Run the [classification review example](../examples/classification.md), inspect [set diagnostics](diagnostics.md), and read the [generated class reference](../classes/conformal-classifier.md).
