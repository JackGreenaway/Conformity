"""Constructor configuration survives sklearn orchestration."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from conformity import ConformalClassifier, ConformalRegressor
from conformity._typing import FeatureMatrix


@pytest.mark.parametrize("classification", [False, True])
def test_constructor_calibration_and_fit_override(classification: bool) -> None:
    """Verify constructor calibration and fit override."""
    if classification:
        X, y = make_classification(n_samples=100, random_state=0)
        model = ConformalClassifier(
            LogisticRegression(max_iter=500), auto_calibrate=True, random_state=0
        )
    else:
        X, y = make_regression(n_samples=100, random_state=0)
        model = ConformalRegressor(Ridge(), auto_calibrate=True, random_state=0)
    fitted = clone(model).fit(X, y)
    assert fitted.n_calibration_ == 20
    fitted.fit(X, y, auto_calibrate=False)
    assert not fitted.is_calibrated_


def test_internal_search_excludes_calibration_from_preprocessing_and_cv() -> None:
    """Verify internal search excludes calibration from preprocessing and cv."""
    X, y = make_regression(n_samples=100, random_state=0)
    search = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1.0]}, cv=3
    )
    model = ConformalRegressor(
        search, auto_calibrate=True, tts_kwargs={"test_size": 0.3}, random_state=0
    )
    model.fit(X, y)
    assert model.n_calibration_ == 30
    scaler = model.estimator_.best_estimator_.named_steps["standardscaler"]
    assert scaler.n_samples_seen_ == 70
    assert not hasattr(search, "best_estimator_")
    assert model.predict_interval(X[:5]).shape == (5, 2)
    assert np.isfinite(model.predict_point(X[:5])).all()
    assert model.tts_kwargs == {"test_size": 0.3}
    model.fit(X, y, tts_kwargs={"test_size": 0.4})
    assert model.n_calibration_ == 40


def test_outer_pipeline_predictions_use_fitted_preprocessing() -> None:
    """Verify outer pipeline predictions use fitted preprocessing."""
    X, y = make_regression(n_samples=100, random_state=0)
    pipe = make_pipeline(
        StandardScaler(), ConformalRegressor(Ridge(), prediction_mode="point")
    )
    pipe.fit(X[:70], y[:70])
    pipe[-1].calibrate(pipe[:-1].transform(X[70:]), y[70:])
    points = pipe.predict(X[:5])
    interval_points, bounds = pipe.predict(X[:5], return_interval=True, alpha=0.1)
    np.testing.assert_allclose(points, interval_points)
    assert bounds.shape == (5, 2)
    np.testing.assert_allclose(pipe.predict(X[:5], return_interval=False), points)


@pytest.mark.parametrize("classification", [False, True])
def test_pipeline_qualified_weights_follow_internal_training_split(
    classification: bool,
) -> None:
    """Verify pipeline qualified weights follow internal training split."""
    from sklearn.model_selection import train_test_split

    if classification:
        X, y = make_classification(n_samples=100, random_state=0)
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        wrapper = ConformalClassifier
        weight_name = "logisticregression__sample_weight"
    else:
        X, y = make_regression(n_samples=100, random_state=0)
        estimator = make_pipeline(StandardScaler(), Ridge())
        wrapper = ConformalRegressor
        weight_name = "ridge__sample_weight"
    weights = np.arange(1, 101, dtype=float)
    model = wrapper(estimator, auto_calibrate=True, random_state=7).fit(
        X, y, **{weight_name: weights}
    )
    train, _ = train_test_split(np.arange(100), test_size=0.2, random_state=7)
    reference = clone(estimator).fit(
        X[train], y[train], **{weight_name: weights[train]}
    )
    np.testing.assert_allclose(model.estimator_[-1].coef_, reference[-1].coef_)
    assert model.n_calibration_ == 20
    assert not hasattr(estimator[-1], "coef_")


def test_outer_classifier_pipeline_and_prediction_override() -> None:
    """Verify outer classifier pipeline and prediction override."""
    X, y = make_classification(n_samples=100, random_state=0)
    pipe = make_pipeline(
        StandardScaler(),
        ConformalClassifier(LogisticRegression(max_iter=500), prediction_mode="point"),
    )
    pipe.fit(X[:70], y[:70])
    pipe[-1].calibrate(pipe[:-1].transform(X[70:]), y[70:])
    labels = pipe.predict(X[:5])
    sets, probabilities = pipe.predict(X[:5], return_set=True, alpha=0.1)
    mask = pipe[-1].predict_set(pipe[:-1].transform(X[:5]), alpha=0.1)
    np.testing.assert_array_equal(~np.isnan(sets), mask)
    np.testing.assert_allclose(probabilities, pipe.predict_proba(X[:5]))
    np.testing.assert_array_equal(labels, pipe.predict(X[:5], return_set=False))
    with pytest.raises(ValueError, match="return_set"):
        pipe.predict(X[:5], return_set="yes")


@pytest.mark.parametrize("weights", [[1], [-1] * 100, [0] * 100, [np.nan] * 100])
def test_invalid_pipeline_qualified_weights_rejected(weights: ArrayLike) -> None:
    """Verify invalid pipeline qualified weights rejected."""
    X, y = make_regression(n_samples=100, random_state=0)
    model = ConformalRegressor(
        make_pipeline(StandardScaler(), Ridge()), auto_calibrate=True
    )
    with pytest.raises(ValueError, match="sample_weight"):
        model.fit(X, y, ridge__sample_weight=weights)


@pytest.mark.parametrize("classification", [False, True])
def test_cross_validate_final_wrapper_auto_calibrates_each_fold(
    classification: bool,
) -> None:
    """Verify cross validate final wrapper auto calibrates each fold."""
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import Pipeline

    if classification:
        X, y = make_classification(n_samples=120, random_state=0)
        inner = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        control = ConformalClassifier(
            inner, auto_calibrate=True, prediction_mode="point", random_state=7
        )
        scoring = "accuracy"
    else:
        X, y = make_regression(n_samples=120, noise=10, random_state=0)
        inner = make_pipeline(StandardScaler(), Ridge())
        control = ConformalRegressor(
            inner, auto_calibrate=True, prediction_mode="point", random_state=7
        )
        scoring = "neg_mean_absolute_error"
    pipe = Pipeline([("conformal", control)])

    def coverage(fitted: Pipeline, X_test: FeatureMatrix, y_test: ArrayLike) -> float:
        """Score held-out conformal coverage for a fitted cross-validation pipeline."""
        return fitted.named_steps["conformal"].evaluate(X_test, y_test, alpha=0.1)[
            "coverage"
        ]

    cv = KFold(n_splits=3, shuffle=True, random_state=11)
    result = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring={"point": scoring, "coverage": coverage},
        return_estimator=True,
        error_score="raise",
    )
    assert np.isfinite(result["test_point"]).all()
    assert ((result["test_coverage"] >= 0) & (result["test_coverage"] <= 1)).all()
    from sklearn.model_selection import train_test_split

    for fitted, (fold_train, fold_test) in zip(result["estimator"], cv.split(X, y)):
        model = fitted.named_steps["conformal"]
        train, calib = train_test_split(
            np.arange(len(fold_train)), test_size=0.2, random_state=7
        )
        assert model.n_calibration_ == len(calib) == 16
        scaler = model.estimator_.named_steps["standardscaler"]
        assert scaler.n_samples_seen_ == len(train) == 64
        np.testing.assert_allclose(scaler.mean_, X[fold_train[train]].mean(axis=0))
        if classification:
            assert model.predict_set(X[fold_test], alpha=0.1).shape == (40, 2)
        else:
            assert model.predict_interval(X[fold_test], alpha=0.1).shape == (40, 2)
    assert not hasattr(control, "estimator_")
    assert len({id(fitted[-1].estimator_) for fitted in result["estimator"]}) == 3


@pytest.mark.parametrize("classification", [False, True])
def test_cross_validate_with_preprocessing_before_final_wrapper(
    classification: bool,
) -> None:
    """API compatibility; this layout does not establish conformal validity."""
    from sklearn.model_selection import KFold, cross_validate

    if classification:
        X, y = make_classification(n_samples=120, random_state=0)
        control = ConformalClassifier(
            LogisticRegression(max_iter=500),
            auto_calibrate=True,
            prediction_mode="point",
            random_state=7,
        )
        scoring = "accuracy"
    else:
        X, y = make_regression(n_samples=120, noise=10, random_state=0)
        control = ConformalRegressor(
            Ridge(), auto_calibrate=True, prediction_mode="point", random_state=7
        )
        scoring = "neg_mean_absolute_error"
    pipe = make_pipeline(StandardScaler(), control)
    result = cross_validate(
        pipe,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=11),
        scoring=scoring,
        return_estimator=True,
        error_score="raise",
    )
    assert np.isfinite(result["test_score"]).all()
    for fitted in result["estimator"]:
        assert fitted[0].n_samples_seen_ == 80
        assert fitted[-1].n_calibration_ == 16
        if classification:
            sets, proba = fitted.predict(X[:5], return_set=True, alpha=0.1)
            assert sets.shape == proba.shape == (5, 2)
        else:
            points, bounds = fitted.predict(X[:5], return_interval=True, alpha=0.1)
            assert points.shape == (5,)
            assert bounds.shape == (5, 2)
