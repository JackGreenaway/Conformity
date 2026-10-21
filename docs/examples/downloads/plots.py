"""Create regression interval and classification prediction set plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conformity import (
    ConformalClassifier,
    ConformalRegressor,
    prediction_set_coverage,
    prediction_set_coverage_by_class,
)

# Compare symmetric and asymmetric intervals using identical data partitions.
rng = np.random.default_rng(31)
X = rng.uniform(20, 500, size=(500, 1))
y = 8 + 0.08 * X[:, 0] + rng.exponential(4, 500) - 4
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=33
)
order = np.argsort(X_test[:, 0])
x = X_test[order, 0]
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharey=True)
for ax, method, title in zip(
    axes,
    ("symmetric", "asymmetric"),
    ("Symmetric: one cutoff", "Asymmetric: separate tail cutoffs"),
):
    regressor = ConformalRegressor(
        LinearRegression(), auto_calibrate=False, interval_method=method
    )
    regressor.fit(X_train, y_train)
    regressor.calibrate(X_calib, y_calib)
    points, intervals = regressor.predict(
        X_test[order], alpha=0.1, return_interval=True
    )
    report = regressor.evaluate(X_test, y_test, alpha=0.1)
    print(
        f"{method}: coverage={report['coverage']:.3f}, "
        f"lower_miss={report['lower_miss_rate']:.3f}, "
        f"upper_miss={report['upper_miss_rate']:.3f}, "
        f"mean_width={report['mean_width']:.2f}, "
        f"normalized_width={report['normalized_mean_width']:.2f}, "
        f"interval_score={report['interval_score']:.2f}"
    )
    ax.fill_between(
        x, intervals[:, 0], intervals[:, 1], alpha=0.25, label="90% interval"
    )
    ax.plot(x, points, color="C1", label="Point prediction")
    ax.scatter(x, y_test[order], s=16, color="C0", label="Held-out outcome")
    ax.set(xlabel="Distance (km)", title=title)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Delivery time (hours)")
fig.tight_layout()
fig.savefig("symmetric-asymmetric-intervals.png", dpi=140)
assert points.shape == (100,)
plt.close(fig)

# Classification: set size in feature space and set size counts.
X, y = make_classification(
    n_samples=900,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.4,
    random_state=41,
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
classifier = ConformalClassifier(
    Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    ),
    method="lac",
    calibration_splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=43),
).fit(X_dev, y_dev)
membership, _ = classifier.predict(X_test, alpha=0.1, return_set=True)
set_sizes = membership.sum(axis=1)
counts = np.bincount(set_sizes, minlength=len(classifier.classes_) + 1)
coverage_by_class = prediction_set_coverage_by_class(
    y_test, membership, classes=classifier.classes_
)
overall_coverage = prediction_set_coverage(
    y_test, membership, classes=classifier.classes_
)
print(f"classification: marginal test coverage={overall_coverage:.3f}")
for label in classifier.classes_:
    class_count = int(np.sum(y_test == label))
    print(
        f"class {label}: observed coverage={coverage_by_class[label]:.3f} "
        f"(n={class_count})"
    )

fig, (scatter_ax, count_ax, coverage_ax) = plt.subplots(3, 1, figsize=(7, 9))
for label, name in enumerate(("Class 0", "Class 1", "Class 2")):
    selected = y_test == label
    scatter_ax.scatter(
        X_test[selected, 0],
        X_test[selected, 1],
        s=24 + 28 * set_sizes[selected],
        alpha=0.7,
        label=name,
    )
scatter_ax.set(
    xlabel="Feature 1", ylabel="Feature 2", title="Held-out samples by true class"
)
scatter_ax.legend(title="Marker size = set size", fontsize=8)
sizes = np.arange(len(counts))
count_ax.bar(sizes, counts, color="C2")
count_ax.set(
    xticks=sizes,
    xlabel="Candidate labels in set",
    ylabel="Held-out samples",
    title="Prediction set sizes",
)
class_labels = [
    f"Overall\n(n={len(y_test)})",
    *[f"{label}\n(n={np.sum(y_test == label)})" for label in classifier.classes_],
]
class_rates = [
    overall_coverage,
    *[coverage_by_class[label] for label in classifier.classes_],
]
coverage_ax.bar(class_labels, class_rates, color="C4")
coverage_ax.axhline(0.9, color="black", linestyle="--", label="90% marginal reference")
coverage_ax.set(
    ylim=(0, 1.05),
    xlabel="Population / true class",
    ylabel="Observed coverage",
    title="Observed test coverage",
)
coverage_ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("classification-prediction-sets.png", dpi=140)
assert membership.shape == (225, 3)
assert counts.sum() == len(X_test)
plt.close(fig)
