# Classification review

## Question

Which labels should remain plausible when a point classifier must choose just one? Use a prediction set to expose ambiguity and measure how often the true label remains in that set.

## Configuration choices

We use string labels to demonstrate column mapping, a fixed synthetic three-class problem, LAC scores and 10% target miscoverage. The scaler belongs inside the wrapped model. The independent test split is stratified for evaluation; the internal calibration split uses the wrapper defaults.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalClassifier, prediction_set_coverage

X, raw_y = make_classification(
    n_samples=900,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    random_state=21,
)
y = np.array(["low", "medium", "high"])[raw_y]
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.25, random_state=22, stratify=y
)
model = ConformalClassifier(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    method="lac",
    calibration_size=0.25,
    random_state=23,
).fit(X_dev, y_dev)
mask, probabilities = model.predict(X_test, alpha=0.1, return_set=True)
labels = [model.classes_[row].tolist() for row in mask]
coverage = prediction_set_coverage(y_test, mask, classes=model.classes_)
assert mask.dtype == bool
assert mask.shape == probabilities.shape == (225, 3)
assert len(labels) == 225
assert 0 <= coverage <= 1
print(model.evaluate(X_test, y_test, alpha=0.1))
print("First three prediction sets:", labels[:3])
```

## Results and interpretation

`return_set=True` returns **(boolean membership, probabilities)**, not (point labels, sets). Columns follow `classes_`, not the order of the input labels. The printed report includes coverage, mean size, empty rate and singleton rate. A singleton contains one candidate, a larger set exposes ambiguity, and an empty set says no candidate passed the threshold. None is a per-observation confidence probability.

## Next steps

Evaluate a review policy on a separate test set. Compare LAC with APS on development data, then reserve fresh calibration data for the selected method. See [classification guidance](../user-guide/classification.md) for deterministic APS limitations.

The [statistical specification](../legacy/THEORY.md#evaluation-and-limits) explains the coverage assumptions and limits behind this example.
