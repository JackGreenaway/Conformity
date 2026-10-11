"""Tests for conformal classifier."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from conformity import ConformalClassifier


@pytest.fixture
def synthetic_classification_data() -> tuple[
    NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
]:
    """Verify synthetic classification data."""
    X, y = make_classification(
        n_samples=10_000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=927,
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_fit_and_calibrate(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify fit and calibrate."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)

    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    assert clf.is_calibrated_
    assert hasattr(clf, "calibration_scores_")
    assert hasattr(clf, "n_calibration_")


def test_predict_without_calibration_raises(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify predict without calibration raises."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)
    clf.fit(X_train, y_train)

    with pytest.raises(RuntimeError):
        clf.predict(X_test, return_set=True)


def test_multiple_calibrations_warn(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify multiple calibrations warn."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)

    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    with warnings.catch_warnings(record=True) as w:
        clf.calibrate(X_test, y_test)

        assert any("already calibrated" in str(warn.message) for warn in w)


def test_auto_calibrate_argument(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify auto calibrate argument."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)

    clf.fit(
        np.concatenate([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        auto_calibrate=True,
        tts_kwargs={"test_size": 0.2, "random_state": 1},
    )

    assert clf.is_calibrated_


def test_calibrate_without_fit_raises(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify calibrate without fit raises."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)

    with pytest.raises((RuntimeError, AttributeError)):
        clf.calibrate(X_test, y_test)


def test_mismatched_x_y_shapes_raises(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify mismatched x y shapes raises."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)

    with pytest.raises(ValueError):
        clf.fit(X_train, y_train[:-10])


def test_calibrate_with_mismatched_shapes_raises(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify calibrate with mismatched shapes raises."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)
    clf.fit(X_train, y_train)

    with pytest.raises((ValueError, IndexError)):
        clf.calibrate(X_test, y_test[:-5])


def test_auto_calibrate_with_small_dataset(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Verify auto calibrate with small dataset."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    # Use smaller subset
    X_small = X_train[:50]
    y_small = y_train[:50]

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)

    clf.fit(
        X_small,
        y_small,
        auto_calibrate=True,
        tts_kwargs={"test_size": 0.3, "random_state": 1},
    )

    assert clf.is_calibrated_
    assert hasattr(clf, "calibration_scores_")


def test_calibration_set_size_recorded(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify calibration set size recorded."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    assert clf.n_calibration_ == X_test.shape[0]


def test_non_conformity_scores_are_valid(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify non conformity scores are valid."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    # Non-conformity scores should be numeric
    assert hasattr(clf, "calibration_scores_")
    assert clf.calibration_scores_ is not None
    assert len(clf.calibration_scores_) > 0
    assert all(isinstance(x, (int, float, np.number)) for x in clf.calibration_scores_)


def test_predict_returns_output(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify predict returns output."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    result = clf.predict(X_test[:10], alpha=0.1, return_set=True)

    assert result is not None


def test_predict_with_different_alpha_values(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify predict with different alpha values."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    result1 = clf.predict(X_test[:10], alpha=0.01, return_set=True)
    result2 = clf.predict(X_test[:10], alpha=0.5, return_set=True)

    assert result1 is not None
    assert result2 is not None


def test_fit_with_different_estimators(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify fit with different estimators."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    from sklearn.tree import DecisionTreeClassifier

    for EstimatorClass in [LogisticRegression, DecisionTreeClassifier]:
        if EstimatorClass == LogisticRegression:
            clf = ConformalClassifier(
                EstimatorClass(max_iter=1000), auto_calibrate=False
            )
        else:
            clf = ConformalClassifier(EstimatorClass(), auto_calibrate=False)

        clf.fit(X_train, y_train)
        clf.calibrate(X_test, y_test)

        assert clf.is_calibrated_


def test_single_sample_prediction_returns(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> None:
    """Verify single sample prediction returns."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)
    clf.fit(X_train, y_train)
    clf.calibrate(X_test, y_test)

    result = clf.predict(X_test[:1], alpha=0.1, return_set=True)

    assert result is not None


def test_fit_preserves_data_shapes(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Verify fit preserves data shapes."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)

    original_X_shape = X_train.shape
    original_y_shape = y_train.shape

    clf.fit(X_train, y_train)

    assert X_train.shape == original_X_shape
    assert y_train.shape == original_y_shape


def test_calibrate_preserves_data_shapes(
    synthetic_classification_data: tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
    ],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Verify calibrate preserves data shapes."""
    X_train, X_test, y_train, y_test = synthetic_classification_data

    clf = ConformalClassifier(LogisticRegression(max_iter=1000), auto_calibrate=False)
    clf.fit(X_train, y_train)

    original_X_shape = X_test.shape
    original_y_shape = y_test.shape

    clf.calibrate(X_test, y_test)

    assert X_test.shape == original_X_shape
    assert y_test.shape == original_y_shape
