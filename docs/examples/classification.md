# Classification review

[Download the complete Python example](downloads/classification.py). Run it with `python classification.py` in the development environment.

## Question

Which team should receive a support ticket: billing, technical support or account services? A point classifier must choose just one label. Use a prediction set to expose ambiguity and measure how often the true label remains in that set.

## Configuration choices

The final-step wrapper keeps scaling separate from classification. We use manual calibration so the scaler and classifier are fitted before either sees calibration rows. `Pipeline` does not forward custom `calibrate` or `evaluate` methods, so those calls use the final wrapper and features transformed by the fitted scaler. If calibration must happen automatically during `fit`, wrap the whole pipeline instead; see [placement choices](../user-guide/estimators.md#why-the-split-matters).

We use a synthetic three-class dataset as a stand-in for six numeric ticket features, string team labels to demonstrate column mapping, LAC scores and 10% target miscoverage. These simulated features are not real ticket text; an actual text workflow would fit its feature extractor only on training data. The scaler precedes a wrapper around the final classifier. The test and calibration splits are random and unstratified, keeping row selection independent of observed outcomes. The test split reserves 225 tickets; we hold out 169 calibration rows from the remaining 675 before fitting the pipeline on 506.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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
y = np.array(["billing", "technical", "account"])[raw_y]
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=22)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=23
)
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "conformal",
            ConformalClassifier(
                LogisticRegression(max_iter=500), method="lac", auto_calibrate=False
            ),
        ),
    ]
).fit(X_train, y_train)
conformal = model.named_steps["conformal"]
conformal.calibrate(model[:-1].transform(X_calib), y_calib)
mask, probabilities = model.predict(X_test, alpha=0.1, return_set=True)
labels = [model.classes_[row].tolist() for row in mask]
coverage = prediction_set_coverage(y_test, mask, classes=model.classes_)
assert mask.dtype == bool
assert mask.shape == probabilities.shape == (225, 3)
assert len(labels) == 225
assert set(model.classes_) == {"billing", "technical", "account"}
needs_review = mask.sum(axis=1) != 1
assert needs_review.shape == (225,)
assert 0 <= coverage <= 1
for metric, value in conformal.evaluate(
    model[:-1].transform(X_test), y_test, alpha=0.1
).items():
    print(f"{metric}: {value:.3f}")
print("First three candidate teams:", labels[:3])
print("Tickets needing review:", int(needs_review.sum()))
```

<!-- BEGIN EXAMPLE OUTPUT -->

```text
accuracy: 0.618
log_loss: 0.907
coverage: 0.898
mean_size: 2.164
empty_rate: 0.000
singleton_rate: 0.129
First three candidate teams: [['account', 'technical'], ['account', 'billing'], ['account', 'technical']]
Tickets needing review: 196
```

<!-- END EXAMPLE OUTPUT -->

## Results and interpretation

This run retains the true team for 202 of 225 test tickets (89.8%), while returning about 2.16 candidates per ticket. The review rule flags 196 tickets: coverage alone does not make this classifier efficient enough to automate most routing decisions. A test sample can fall below 90% without contradicting the population marginal guarantee.

`return_set=True` returns **(boolean membership, probabilities)**, not (point labels, sets). Columns follow `classes_`, not the order of the input labels. The printed report includes coverage, mean size, empty rate and singleton rate. A singleton contains one candidate, a larger set exposes ambiguity, and an empty set says no candidate passed the threshold. The illustrative review rule flags both empty and multiple-team sets; a singleton names one candidate team. None is a per-observation confidence probability or a guaranteed error rate for automatically routed tickets.

## Next steps

Evaluate a review policy on a separate test set. Compare LAC with APS on development data, then reserve fresh calibration data for the selected method. See [classification guidance](../user-guide/classification.md) for deterministic APS limitations.

The [statistical specification](../legacy/THEORY.md#evaluation-and-limits) explains the coverage assumptions and limits behind this example.
