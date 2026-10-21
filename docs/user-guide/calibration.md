# Calibration and guarantees

## Which conformal method does Conformity implement?

Conformity implements **split conformal prediction (inductive conformal prediction)**. Its automatic workflow splits the supplied data once: it fits the estimator on the training partition and calibrates the fitted model on the held-out partition. The default splitter is `ShuffleSplit(n_splits=1, test_size=0.2)`; pass a different compatible one-split splitter as `calibration_splitter` to control the partition. This is still split conformal prediction, regardless of which splitter you choose.

You can also manage that same split yourself: set `auto_calibrate=False`, call `fit(X_train, y_train)`, then `calibrate(X_calibration, y_calibration)`. Keep final test data separate for evaluation. See [automatic calibration splits](#choosing-a-calibration-size) and [manual calibration](estimators.md#manual-calibration-with-the-complete-pipeline).

Conformity does **not** implement full conformal prediction (transductive conformal prediction). A splitter cannot turn the current API into full conformal: that method must append each candidate outcome to the reference sample, refit the estimator, score the entire augmented sample symmetrically, and calculate a candidate-specific rank p-value. The [theory guide](../theory.md#split-conformal-prediction-inductive-and-full-conformal-prediction-transductive) gives the algorithm sketch and discusses the cost and limits. Both methods' usual finite-sample marginal guarantee relies on exchangeability; chronological splitting alone does not establish it for dependent or drifting time series.

## Question: what does a 90% interval promise?

Under exchangeability, an independently fitted score function gives marginal coverage at least $1-\alpha$ for a fresh observation. It does not give a 90% probability statement for an already observed interval or guarantee coverage for a selected subgroup.

With $n$ held-out scores, Conformity uses the rank

$$
k = \lceil (n+1)(1-\alpha) \rceil.
$$

If $k \le n$, the threshold is the $k$th sorted score; otherwise it is infinite. Score ties are included. Do not replace the corrected rank with a plain interpolated empirical quantile.

## Choosing a calibration size

For a wrapper on the final pipeline model, reserve calibration before fitting any learned preprocessing, set `auto_calibrate=False`, then calibrate on transformed held-out features. For automatic calibration with learned preprocessing, wrap the entire pipeline so its internal split precedes every fitting step. The [pipeline guide](estimators.md#why-the-split-matters) explains why these workflows differ.

A larger calibration partition gives a finer score quantile but leaves fewer fitting observations. Configure it with `calibration_splitter=ShuffleSplit(n_splits=1, test_size=0.25)`; this uses scikit-learn's `test_size` fraction or sample-count semantics. By default, automatic calibration uses `ShuffleSplit(n_splits=1, test_size=0.2)`, with no fixed random seed. Pass a different one-split splitter to choose a random seed or split strategy. The splitter must yield exactly one pair of disjoint train and calibration indices. A reproducible split does not make temporal, grouped or shifted observations exchangeable.

## Classification with rare classes

`ConformalClassifier` enables `warn_on_rare_classes=True` by default. After an automatic split, it warns if any class in `y` is missing from either the model-training or calibration partition. The warning identifies the missing class and suggests stratification; it does not change the split. A class missing from training may be impossible for the estimator to learn, while a class missing from calibration has no held-out scores in this calibration sample.

When a class has enough examples for both partitions, `StratifiedShuffleSplit` can help preserve its representation:

```python
import warnings

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from conformity import ConformalClassifier

X, y = make_classification(
    n_samples=500,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    weights=[0.98, 0.02],
    random_state=7,
)

# The default random split can put all ten minority examples in training.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    random_model = ConformalClassifier(
        LogisticRegression(),
        calibration_splitter=ShuffleSplit(n_splits=1, test_size=0.2, random_state=2),
    ).fit(X, y)
print("Default split warning:", caught[0].message)

# Stratification keeps this class represented in both partitions.
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=11)
model = ConformalClassifier(LogisticRegression(), calibration_splitter=splitter).fit(
    X, y
)
assert model.n_calibration_ == 100
mask, probabilities = model.predict(X[:3], return_set=True)
candidate_sets = [model.classes_[row].tolist() for row in mask]
print("Class columns:", model.classes_.tolist())
print("Calibration rows:", model.n_calibration_)
print("First three prediction sets:", candidate_sets)
print("First probability row:", probabilities[0].round(3).tolist())
```

Example output:

```text
Default split warning: Automatic calibration split omits observed class(es) from calibration partition: [1]. Consider StratifiedShuffleSplit as calibration_splitter when each class has enough observations for both partitions. Set warn_on_rare_classes=False to silence this warning.
Class columns: [0, 1]
Calibration rows: 100
First three prediction sets: [[0], [0], [0]]
First probability row: [0.996, 0.004]
```

To silence the omission warning for a workflow where it is not useful, set `warn_on_rare_classes=False` on the classifier. Stratification cannot put a class in both partitions when there are too few examples. It is an engineering safeguard, not a class-wise coverage guarantee; the standard finite-sample guarantee is for the default random split under exchangeability, not a label-stratified split. For a formal class-wise guarantee, use a method that calibrates each class separately. See the [classification guide](classification.md#rare-classes-and-automatic-calibration) for more detail.

## Supplying a custom automatic-calibration split

`calibration_splitter` follows the scikit-learn splitter interface: it has a `split(X, y)` method that yields training and calibration row indices. Conformity calls it once and requires exactly one split. This lets you encode domain rules in a small splitter while keeping the usual `fit(X, y)` workflow.

For example, the splitter below holds out whole groups. Since Conformity calls `split(X, y)` without a `groups` argument, the group labels are stored in the splitter and checked against the current input length:

```python
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


class StoredGroupSplit:
    def __init__(self, groups, *, test_size=0.2, random_state=None):
        self.groups = np.asarray(groups)
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None):
        if len(self.groups) != len(X):
            raise ValueError("groups must match the rows passed to fit")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield from splitter.split(X, y, groups=self.groups)


groups = np.repeat(np.arange(20), 5)
splitter = StoredGroupSplit(groups, test_size=0.2, random_state=7)
```

Pass it as `ConformalRegressor(estimator, calibration_splitter=splitter)` when fitting on the same rows used to define `groups`. This stored-label example is not directly suitable for an outer CV loop: each fold fits on a subset, but Conformity's current `fit` API does not route groups to the splitter. For grouped cross-validation, precompute each fold's train/calibration indices and call `fit`/`calibrate` explicitly, or extend the wrapper API to accept and route `groups`. Grouping prevents the same group appearing on both sides of a split; it does not by itself guarantee that group-level scores and future scores are exchangeable. See scikit-learn's [`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) documentation.

## Time-ordered data

For temporal data, use earlier observations to fit and a later contiguous window to calibrate. Do not use the default shuffled split: nearby observations can leak information across the split, and a shuffled holdout does not represent forecasting into the future. If the calibration boundary is already known and you do not need a gap, pass scikit-learn's `PredefinedSplit` directly:

```python
import numpy as np
from sklearn.model_selection import PredefinedSplit

# Replace this ordered array with your time-ordered feature matrix.
X = np.empty((100, 1))
n_samples = len(X)
calibration_start = n_samples - 15
test_fold = np.full(n_samples, -1, dtype=int)
test_fold[calibration_start:] = 0
splitter = PredefinedSplit(test_fold)
```

Entries marked `-1` are always in the training partition; the one block marked `0` becomes calibration. `PredefinedSplit` implements the required `.split(X, y)` interface and yields one split for this setup. See the [scikit-learn reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html).

For a computed boundary, purge gap, or a split chosen from `TimeSeriesSplit`, use a small adapter. `TimeSeriesSplit` itself yields multiple folds and requires at least two, whereas Conformity automatic calibration accepts exactly one. The adapter below selects its final fold:

```python
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from conformity import ConformalRegressor


class LastTimeSeriesSplit(BaseEstimator):
    """Use the final TimeSeriesSplit fold as the calibration partition."""

    def __init__(self, n_splits=2, test_size=15, gap=3):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None):
        cv = TimeSeriesSplit(
            n_splits=self.n_splits, test_size=self.test_size, gap=self.gap
        )
        yield list(cv.split(X, y))[-1]


# X and y must already be ordered by time. Keep a later block for final testing.
X = np.arange(120, dtype=float).reshape(-1, 1)
y = 0.5 * X[:, 0] + np.sin(X[:, 0] / 5)
X_dev, y_dev = X[:100], y[:100]
X_future, y_future = X[100:], y[100:]

model = ConformalRegressor(
    Ridge(),
    calibration_splitter=LastTimeSeriesSplit(n_splits=2, test_size=15, gap=3),
).fit(X_dev, y_dev)
points, intervals = model.predict(X_future, return_interval=True)
```

Here the wrapper fits Ridge on the rows before the gap, calculates residual scores on the next 15 rows, and leaves the final 20 rows untouched for future evaluation. `gap` should reflect the forecast horizon, feature lookahead, and any overlap in how targets are constructed; it is not a universal fixed value. The example's final-fold adapter is sklearn-cloneable, so it can also be used as a constructor parameter in a cloned estimator.

Chronological splitting prevents training on future rows when calibrating an earlier window. It does **not** make standard split-conformal coverage valid for autocorrelated, nonstationary, or drifting series: the usual rank guarantee assumes calibration scores and the future score are exchangeable conditional on the fitted model. Treat the intervals as a useful empirical calibration unless your temporal assumptions justify a formal guarantee. Evaluate on later rolling windows, and use a time-series or online conformal method with an appropriate guarantee when formal coverage under temporal dependence is required. Scikit-learn also cautions that ordinary shuffled and k-fold validation can be unsuitable for time-series evaluation; see its [time-series cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data).

For finite regression bounds you need at least $\alpha \ge 1/(n+1)$. Selecting alpha after viewing final calibration outcomes undermines the fixed-procedure interpretation. Repeated predictions can reuse stored scores without refitting, but their errors need not be independent.

## Limitations and next steps

Coverage is unweighted and marginal. Learned preprocessing, model tuning and method selection must stay independent of final calibration. Weighted metrics affect evaluation only. For drift, time series, covariate-conditioned guarantees or adaptive online calibration, this package's current methods are insufficient on their own.

Start with [the theory guide](../theory.md) for the intuition and coverage proof. Evaluate coverage and efficiency together using [diagnostics](diagnostics.md).
