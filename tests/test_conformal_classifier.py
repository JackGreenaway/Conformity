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
        n_samples=10_000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=927,
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


def test_calibrate_without_fit_raises(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    with pytest.raises((RuntimeError, AttributeError)):
        clf.calibrate(X_test, y_test)


def test_mismatched_x_y_shapes_raises(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    with pytest.raises(ValueError):
        clf.fit(X_train, y_train[:-10])


def test_calibrate_with_mismatched_shapes_raises(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)

    with pytest.raises((ValueError, IndexError)):
        clf.calibrate(X_test, y_test[:-5])


def test_auto_calibrate_with_small_dataset(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    # Use smaller subset
    X_small = X_train[:50]
    y_small = y_train[:50]

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    clf.fit(
        X_small,
        y_small,
        auto_calibrate=True,
        tts_kwargs={"test_size": 0.3, "random_state": 1},
    )

    assert clf.is_calibrated_
    assert hasattr(clf, "calibration_non_conformity")


def test_calibration_set_size_recorded(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    assert clf.n_calib == X_test.shape[0]


def test_non_conformity_scores_are_valid(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    # Non-conformity scores should be numeric
    assert hasattr(clf, "calibration_non_conformity")
    assert clf.calibration_non_conformity is not None
    assert len(clf.calibration_non_conformity) > 0
    assert all(
        isinstance(x, (int, float, np.number)) for x in clf.calibration_non_conformity
    )


def test_predict_returns_output(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    result = clf.predict(X_test[:10], alpha=0.1)

    assert result is not None


def test_predict_with_different_alpha_values(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    result1 = clf.predict(X_test[:10], alpha=0.01)
    result2 = clf.predict(X_test[:10], alpha=0.5)

    assert result1 is not None
    assert result2 is not None


def test_fit_with_different_estimators(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    from sklearn.tree import DecisionTreeClassifier

    for EstimatorClass in [LogisticRegression, DecisionTreeClassifier]:
        if EstimatorClass == LogisticRegression:
            clf = ConformalClassifier(EstimatorClass(max_iter=1000))
        else:
            clf = ConformalClassifier(EstimatorClass())

        clf.fit(X_train, y_train)
        clf.calibrate(X_test, y_test)

        assert clf.is_calibrated_


def test_single_sample_prediction_returns(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    result = clf.predict(X_test[:1], alpha=0.1)

    assert result is not None


def test_fit_preserves_data_shapes(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))

    original_X_shape = X_train.shape
    original_y_shape = y_train.shape

    clf.fit(X_train, y_train)

    assert X_train.shape == original_X_shape
    assert y_train.shape == original_y_shape


def test_calibrate_preserves_data_shapes(synthetic_classification_data):
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)

    original_X_shape = X_test.shape
    original_y_shape = y_test.shape

    clf.calibrate(X_test, y_test)

    assert X_test.shape == original_X_shape
    assert y_test.shape == original_y_shape
