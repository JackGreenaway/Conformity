"""Regression tests for finite-sample validity and sklearn integration."""

import pickle

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from conformity import (
    ConformalClassifier,
    ConformalRegressor,
    interval_score,
    prediction_interval_coverage,
    prediction_interval_width,
    prediction_set_coverage,
    prediction_set_size,
    prediction_set_empty_rate,
    prediction_set_singleton_rate,
)


def regressor():
    reg = ConformalRegressor(DummyRegressor(strategy="constant", constant=0))
    reg.fit(np.zeros((10, 1)), np.zeros(10))
    return reg


class FixedClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.asarray(X, dtype=float)

    def predict(self, X):
        return self.classes_[np.argmax(X, axis=1)]


def classifier(method="lac", labels=("cat", "dog")):
    clf = ConformalClassifier(FixedClassifier(), method=method)
    clf.fit([[0.8, 0.2], [0.2, 0.8]], labels)
    return clf


def test_exact_order_statistic_and_cached_scores():
    reg = regressor().calibrate(np.zeros((9, 1)), np.arange(1, 10))
    intervals = reg.predict_interval([[0]], alpha=0.2)
    np.testing.assert_array_equal(intervals, [[-8, 8]])
    assert reg.q_level_ == 8
    with pytest.raises(ValueError):
        reg.calibration_scores_[0] = 100
    assert reg.n_calibration_ == 9


def test_small_calibration_returns_unbounded_interval():
    reg = regressor().calibrate([[0]], [1])
    with pytest.warns(UserWarning, match="infinite"):
        bounds = reg.predict_interval([[0]], alpha=0.1)
    np.testing.assert_array_equal(bounds, [[-np.inf, np.inf]])
    assert prediction_interval_coverage([100000], bounds) == 1
    assert prediction_interval_width(bounds) == np.inf
    assert interval_score([1], bounds) == np.inf


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf, True, "0.1", [0.1]])
@pytest.mark.parametrize("kind", ["regression", "classification"])
def test_invalid_alpha_rejected(alpha, kind):
    if kind == "regression":
        obj = regressor().calibrate(np.zeros((10, 1)), np.arange(10))
        X = [[0]]
        call = obj.predict_interval
    else:
        obj = classifier().calibrate([[0.8, 0.2]] * 10, ["cat"] * 10)
        X = [[0.8, 0.2]]
        call = obj.predict_set
    with pytest.raises(ValueError, match="alpha"):
        call(X, alpha)


def test_refitting_invalidates_calibration_and_parameter_changes_invalidate_fit():
    reg = regressor().calibrate(np.zeros((10, 1)), np.arange(10))
    reg.fit([[0], [1]], [0, 1])
    assert not reg.is_calibrated_
    assert not hasattr(reg, "n_calib")
    with pytest.raises(RuntimeError, match="calibrated"):
        reg.predict_interval([[0]])
    reg.set_params(estimator__constant=1)
    with pytest.raises(NotFittedError):
        reg.predict_point([[0]])


def test_failed_refit_cannot_reuse_old_calibration():
    reg = regressor().calibrate(np.zeros((10, 1)), np.arange(10))
    with pytest.raises(ValueError):
        reg.fit([[0]], [1, 2])
    assert not reg.is_calibrated_
    with pytest.raises(NotFittedError):
        reg.predict_point([[0]])


@pytest.mark.parametrize("labels", [("cat", "dog"), (-10, 20)])
def test_arbitrary_labels_ties_and_p_values(labels):
    clf = classifier(labels=labels)
    clf.calibrate([[0.8, 0.2]] * 9, [labels[0]] * 9)
    X = [[0.8, 0.2], [0.2, 0.8]]
    mask = clf.predict_set(X, alpha=0.2)
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])
    np.testing.assert_array_equal(clf.predict_p_values(X) > 0.2, mask)
    assert prediction_set_coverage(labels, mask, classes=clf.classes_) == 1
    assert prediction_set_size(mask) == 1
    assert prediction_set_empty_rate(mask) == 0
    assert prediction_set_singleton_rate(mask) == 1
    legacy, proba = clf.predict(X, alpha=0.2)
    assert prediction_set_coverage(list(labels), legacy) == 1
    assert proba.shape == (2, 2)


def test_unknown_calibration_label_rejected_without_losing_valid_scores():
    clf = classifier().calibrate([[0.8, 0.2]] * 9, ["cat"] * 9)
    before = clf.calibration_scores_.copy()
    with pytest.raises(ValueError, match="absent"):
        clf.calibrate([[0.8, 0.2]], ["other"])
    np.testing.assert_array_equal(before, clf.calibration_scores_)


def test_aps_scores_are_cumulative_and_method_change_rejected():
    clf = classifier(method="aps").calibrate([[0.8, 0.2]] * 9, ["dog"] * 9)
    np.testing.assert_allclose(clf.calibration_scores_, 1)
    assert clf.predict_set([[0.8, 0.2]], alpha=0.2).all()
    clf.method = "lac"
    with pytest.raises(ValueError, match="method changed"):
        clf.predict_set([[0.8, 0.2]], alpha=0.2)


def test_classification_unbounded_threshold_includes_all_classes():
    clf = classifier().calibrate([[0.8, 0.2]], ["cat"])
    with pytest.warns(UserWarning, match="infinite"):
        assert clf.predict_set([[0.1, 0.9]], alpha=0.1).all()


@pytest.mark.parametrize(
    "proba", [[[np.nan, 0.2]], [[-0.2, 1.2]], [[0.2, 0.2]], [[0.2, 0.3, 0.5]]]
)
def test_malformed_estimator_probabilities_rejected(proba):
    clf = classifier()
    with pytest.raises(ValueError):
        clf.predict_proba(proba)


@pytest.mark.parametrize("y", [[np.nan], [np.inf], [1, 2]])
def test_invalid_calibration_targets(y):
    with pytest.raises(ValueError):
        regressor().calibrate([[0]], y)


def test_sparse_auto_calibration_and_original_estimator_untouched():
    X = sparse.csr_matrix(np.arange(120, dtype=float).reshape(40, 3))
    y = np.arange(40, dtype=float)
    estimator = LinearRegression()
    reg = ConformalRegressor(estimator, random_state=7, calibration_size=0.25)
    reg.fit(X, y, auto_calibrate=True, sample_weight=np.ones(40))
    assert reg.n_calibration_ == 10
    assert not hasattr(estimator, "coef_")
    assert reg.predict_interval(X[:2], alpha=0.2).shape == (2, 2)
    assert not clone(reg).is_calibrated_
    restored = pickle.loads(pickle.dumps(reg))
    np.testing.assert_allclose(
        restored.predict_interval(X[:2], 0.2), reg.predict_interval(X[:2], 0.2)
    )


def test_reproducible_split_and_training_weight_alignment():
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = np.sin(X[:, 0])
    weight = np.arange(1, 101)
    objects = [
        ConformalRegressor(LinearRegression(), random_state=42) for _ in range(2)
    ]
    for obj in objects:
        obj.fit(X, y, auto_calibrate=True, sample_weight=weight)
    np.testing.assert_allclose(
        objects[0].calibration_scores_, objects[1].calibration_scores_
    )
    from sklearn.model_selection import train_test_split

    train, _ = train_test_split(np.arange(100), test_size=0.2, random_state=42)
    reference = LinearRegression().fit(X[train], y[train], sample_weight=weight[train])
    np.testing.assert_allclose(reference.coef_, objects[0].estimator_.coef_)


def test_sklearn_scoring_and_grid_search():
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = X[:, 0] * 2
    reg = ConformalRegressor(
        make_pipeline(StandardScaler(), LinearRegression()), prediction_mode="point"
    )
    assert is_regressor(reg)
    assert np.all(cross_val_score(reg, X, y, scoring="neg_mean_squared_error") > -1e-20)
    search = GridSearchCV(
        reg, {"estimator__linearregression__fit_intercept": [True, False]}, scoring="r2"
    )
    search.fit(X, y)
    assert search.best_score_ == pytest.approx(1)
    clf = ConformalClassifier(LogisticRegression(), prediction_mode="point")
    assert is_classifier(clf)
    labels = np.tile(["cat", "dog"], 50)
    assert np.isfinite(cross_val_score(clf, X, labels, scoring="neg_log_loss")).all()


@pytest.mark.parametrize(
    "obj",
    [
        ConformalRegressor(LinearRegression(), prediction_mode="point"),
        ConformalClassifier(LogisticRegression(), prediction_mode="point"),
    ],
)
def test_sklearn_estimator_contract(obj):
    check_estimator(obj)


def test_evaluate_and_weighted_metrics():
    reg = regressor().calibrate(np.zeros((19, 1)), np.arange(19))
    result = reg.evaluate([[0], [1]], [0, 100], alpha=0.2, sample_weight=[1, 0])
    assert result["coverage"] == 1
    assert result["mae"] == 0
    assert set(result) == {
        "r2",
        "mae",
        "mse",
        "coverage",
        "mean_width",
        "interval_score",
    }
    clf = classifier().calibrate([[0.8, 0.2]] * 19, ["cat"] * 19)
    result = clf.evaluate([[0.8, 0.2], [0.2, 0.8]], ["cat", "dog"], alpha=0.2)
    assert result["coverage"] == 1
    assert result["accuracy"] == 1


def test_winkler_score_penalizes_missed_bounds():
    assert interval_score([0, 5, -3], [[-1, 1]] * 3, alpha=0.2) == pytest.approx(22)
    assert (
        prediction_interval_coverage([0, 5, -3], [[-1, 1]] * 3, sample_weight=[1, 0, 0])
        == 1
    )


@pytest.mark.parametrize(
    "intervals", [[], [[1, 0]], [[np.nan, 1]], [[0, 1, 2]], [[np.inf, np.inf]]]
)
def test_invalid_intervals_rejected(intervals):
    with pytest.raises(ValueError):
        prediction_interval_width(intervals)


@pytest.mark.parametrize("weights", [[1], [-1, 1], [0, 0], [np.nan, 1]])
def test_invalid_metric_weights(weights):
    with pytest.raises(ValueError):
        prediction_interval_coverage([0, 0], [[-1, 1]] * 2, sample_weight=weights)


def test_boolean_sets_require_class_order():
    with pytest.raises(ValueError, match="classes"):
        prediction_set_coverage(["a"], [[True, False]])


def test_zero_weight_unbounded_interval_ignored():
    assert (
        prediction_interval_width([[0, 1], [-np.inf, np.inf]], sample_weight=[1, 0])
        == 1
    )


def test_dataframe_column_selecting_pipeline():
    pd = pytest.importorskip("pandas")
    from sklearn.compose import ColumnTransformer

    X = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) ** 2})
    y = X.a.to_numpy() * 2
    pipeline = make_pipeline(
        ColumnTransformer([("a", StandardScaler(), ["a"])]), LinearRegression()
    )
    reg = ConformalRegressor(pipeline, random_state=0).fit(X, y, auto_calibrate=True)
    assert reg.predict_interval(X.iloc[:2], alpha=0.2).shape == (2, 2)
    with pytest.raises(ValueError, match="feature names"):
        reg.predict_point(X[["b", "a"]])


def test_large_integer_legacy_labels_keep_precision():
    labels = (2**60 + 1, 2**60 + 2)
    clf = classifier(labels=labels).calibrate([[0.8, 0.2]] * 9, [labels[0]] * 9)
    sets, _ = clf.predict([[0.8, 0.2], [0.2, 0.8]], alpha=0.2)
    assert sets[0, 0] == labels[0]
    assert sets[1, 1] == labels[1]
    assert prediction_set_coverage(labels, sets) == 1


def test_single_class_evaluation():
    from sklearn.dummy import DummyClassifier

    clf = ConformalClassifier(DummyClassifier()).fit([[0]] * 10, ["only"] * 10)
    clf.calibrate([[1]] * 19, ["only"] * 19)
    report = clf.evaluate([[2]], ["only"], alpha=0.2)
    assert report["log_loss"] == 0
    assert report["coverage"] == 1
    assert report["mean_size"] == 1
