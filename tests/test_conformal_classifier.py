import numpy as np
import pytest
import warnings

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from conformity.classifier import ConformalClassifier


@pytest.fixture
def synthetic_classification_data():
    X, y = make_classification(
        n_samples=10_000, n_features=7, n_classes=3, n_informative=3, random_state=927
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_fit_and_calibrate(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    assert clf.is_calibrated_
    assert hasattr(clf, "calibration_non_conformity")
    assert hasattr(clf, "n_calib")


def test_predict_set_shape(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    pred_set, boolean_set, y_prob, q_level = clf.predict(X_test, alpha=0.1)

    assert pred_set.shape == (X_test.shape[0], y_prob.shape[1])
    assert boolean_set.shape == (X_test.shape[0], y_prob.shape[1])
    assert y_prob.shape[0] == X_test.shape[0]
    assert isinstance(q_level, float)


def test_predict_without_calibration_raises(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    clf.fit(X_train, y_train)

    with pytest.raises(RuntimeError):
        clf.predict(X_test)


def test_multiple_calibrations_warn(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    with warnings.catch_warnings(record=True) as w:
        clf.calibrate(X_test, y_test)

        assert any("already calibrated" in str(warn.message) for warn in w)


def test_predict_with_different_alpha(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    pred_set_01, _, _, _ = clf.predict(X_test, alpha=0.01)
    pred_set_20, _, _, _ = clf.predict(X_test, alpha=0.20)

    # With higher alpha, prediction sets should be smaller or equal in size
    size_01 = np.sum(~np.isnan(pred_set_01), axis=1)
    size_20 = np.sum(~np.isnan(pred_set_20), axis=1)

    assert np.all(size_01 >= size_20)


def test_auto_calibrate_argument(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    clf.fit(
        np.concatenate([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        auto_calibrate=True,
        tts_kwargs={"test_size": 0.2, "random_state": 1},
    )

    assert clf.is_calibrated_


def test_prediction_set_contains_true_label(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    pred_set, _, _, _ = clf.predict(X_test, alpha=0.1)

    # Check that at least 80% of true labels are in the prediction set (since alpha=0.1, expect ~90%)
    contained = []
    for i, label in enumerate(y_test):
        contained.append(label in pred_set[i][~np.isnan(pred_set[i])])

    assert np.mean(contained) > 0.8


def test_extreme_inputs():
    X = np.array([[1e10, 0], [0, 1e10], [-1e10, 0], [0, -1e10]])
    y = np.array([0, 1, 0, 1])

    clf = ConformalClassifier(LogisticRegression())

    clf.fit(X, y)
    clf.calibrate(X, y)

    pred_set, boolean_set, y_prob, q_level = clf.predict(X)

    assert pred_set.shape == (4, 2)
    assert boolean_set.shape == (4, 2)
    assert y_prob.shape == (4, 2)
