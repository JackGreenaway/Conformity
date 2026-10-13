"""Run the synthetic classification prediction set example."""

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
model = ConformalClassifier(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=500)),
        ]
    ),
    method="lac",
    tts_kwargs={"random_state": 23},
).fit(X_dev, y_dev)
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
for metric, value in model.evaluate(X_test, y_test, alpha=0.1).items():
    print(f"{metric}: {value:.3f}")
print("First three candidate teams:", labels[:3])
print("Tickets needing review:", int(needs_review.sum()))
