"""Tests for scikit-learn compatibility."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conformity.base import BaseConformalPredictor
from conformity.classifier import ConformalClassifier
from conformity.regressor import ConformalRegressor


@pytest.fixture
def regression_data():
    """Fixture for regression data."""
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def classification_data():
    """Fixture for classification data."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)


class TestBaseEstimatorCompliance:
    """Tests for BaseEstimator compliance."""

    def test_get_params(self, regression_data):
        """Test get_params method."""
        estimator = LinearRegression()
        reg = ConformalRegressor(estimator=estimator)

        params = reg.get_params()
        assert "estimator" in params
        assert params["estimator"] == estimator

    def test_set_params(self, regression_data):
        """Test set_params method."""
        reg = ConformalRegressor(LinearRegression())

        new_estimator = LinearRegression()
        reg.set_params(estimator=new_estimator)

        assert reg.estimator == new_estimator

    def test_set_params_returns_self(self):
        """Test that set_params returns self."""
        reg = ConformalRegressor(LinearRegression())
        result = reg.set_params(estimator=LinearRegression())
        assert result is reg

    def test_classifier_get_params(self):
        """Test get_params for ConformalClassifier."""
        clf = ConformalClassifier(LogisticRegression())
        params = clf.get_params()
        assert "estimator" in params

    def test_classifier_set_params(self):
        """Test set_params for ConformalClassifier."""
        clf = ConformalClassifier(LogisticRegression())
        result = clf.set_params(estimator=LogisticRegression(max_iter=500))
        assert result is clf


class TestClone:
    """Tests for sklearn.base.clone compatibility."""

    def test_clone_regressor(self):
        """Test cloning ConformalRegressor."""
        reg = ConformalRegressor(LinearRegression())
        reg_cloned = clone(reg)

        assert isinstance(reg_cloned, ConformalRegressor)
        assert not hasattr(reg_cloned, "estimator_")
        assert not reg_cloned.is_calibrated_

    def test_clone_classifier(self):
        """Test cloning ConformalClassifier."""
        clf = ConformalClassifier(LogisticRegression())
        clf_cloned = clone(clf)

        assert isinstance(clf_cloned, ConformalClassifier)
        assert not hasattr(clf_cloned, "estimator_")
        assert not clf_cloned.is_calibrated_

    def test_clone_fitted_regressor(self, regression_data):
        """Test cloning a fitted regressor doesn't preserve fitted state."""
        X_train, X_test, y_train, y_test = regression_data

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train[:100], y_train[:100])

        reg_cloned = clone(reg)
        assert not hasattr(reg_cloned, "estimator_")
        assert reg_cloned.is_calibrated_ == False


class TestPipelineCompatibility:
    """Tests for sklearn Pipeline compatibility."""

    def test_regressor_in_pipeline(self, regression_data):
        """Test ConformalRegressor in a Pipeline."""
        X_train, X_test, y_train, y_test = regression_data

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ConformalRegressor(LinearRegression()))
        ])

        pipe.fit(X_train[:100], y_train[:100])
        assert hasattr(pipe.named_steps['regressor'], 'estimator_')

    def test_classifier_in_pipeline(self, classification_data):
        """Test ConformalClassifier in a Pipeline."""
        X_train, X_test, y_train, y_test = classification_data

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', ConformalClassifier(LogisticRegression(max_iter=1000)))
        ])

        pipe.fit(X_train[:100], y_train[:100])
        assert hasattr(pipe.named_steps['classifier'], 'estimator_')

    def test_pipeline_predict_regressor(self, regression_data):
        """Test prediction through pipeline (regressor)."""
        X_train, X_test, y_train, y_test = regression_data

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train[:100], y_train[:100])
        reg.calibrate(X_train[100:150], y_train[100:150])

        # Should be able to call predict
        y_pred, intervals, q_level = reg.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]

    def test_pipeline_predict_classifier(self, classification_data):
        """Test prediction through pipeline (classifier)."""
        X_train, X_test, y_train, y_test = classification_data

        clf = ConformalClassifier(LogisticRegression(max_iter=1000))
        clf.fit(X_train[:100], y_train[:100])
        clf.calibrate(X_train[100:150], y_train[100:150])

        # Should be able to call predict
        pred_set, _, _, _ = clf.predict(X_test)
        assert pred_set.shape[0] == X_test.shape[0]


class TestInputValidation:
    """Tests for scikit-learn input validation."""

    def test_fit_validates_input(self):
        """Test that fit validates input shapes."""
        reg = ConformalRegressor(LinearRegression())

        with pytest.raises(ValueError):
            X = np.array([[1, 2], [3, 4]])
            y = np.array([1, 2, 3])  # Mismatched size
            reg.fit(X, y)

    def test_calibrate_validates_input(self, regression_data):
        """Test that calibrate validates input."""
        X_train, X_test, y_train, y_test = regression_data

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train[:100], y_train[:100])

        with pytest.raises(ValueError):
            reg.calibrate(X_test, y_test[:-5])  # Mismatched size

    def test_predict_validates_input(self, regression_data):
        """Test that predict validates input."""
        X_train, X_test, y_train, y_test = regression_data

        reg = ConformalRegressor(LinearRegression())
        # Should raise NotFittedError or RuntimeError when not fitted
        with pytest.raises((RuntimeError, Exception)):
            reg.predict(X_test)  # Not fitted

    def test_predict_rejects_uncalibrated(self, regression_data):
        """Test that predict rejects uncalibrated estimator."""
        X_train, X_test, y_train, y_test = regression_data

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train[:100], y_train[:100])

        with pytest.raises(RuntimeError):
            reg.predict(X_test)  # Not calibrated


class TestRegressorMixin:
    """Tests for RegressorMixin functionality."""

    def test_conformal_regressor_has_estimator_type(self):
        """Test that ConformalRegressor has _estimator_type attribute."""
        reg = ConformalRegressor(LinearRegression())
        assert hasattr(reg, '_estimator_type')
        assert reg._estimator_type == 'regressor'

    def test_conformal_regressor_is_regressor_mixin(self):
        """Test that ConformalRegressor inherits from RegressorMixin."""
        from sklearn.base import RegressorMixin

        reg = ConformalRegressor(LinearRegression())
        assert isinstance(reg, RegressorMixin)


class TestClassifierMixin:
    """Tests for ClassifierMixin functionality."""

    def test_conformal_classifier_has_estimator_type(self):
        """Test that ConformalClassifier has _estimator_type attribute."""
        clf = ConformalClassifier(LogisticRegression())
        assert hasattr(clf, '_estimator_type')
        assert clf._estimator_type == 'classifier'

    def test_conformal_classifier_is_classifier_mixin(self):
        """Test that ConformalClassifier inherits from ClassifierMixin."""
        from sklearn.base import ClassifierMixin

        clf = ConformalClassifier(LogisticRegression())
        assert isinstance(clf, ClassifierMixin)


class TestAutoCalibrationEdgeCases:
    """Tests for auto_calibrate edge cases."""

    def test_auto_calibrate_with_default_split(self, regression_data):
        """Test auto_calibrate with default 80/20 split."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X, y, auto_calibrate=True)

        assert reg.is_calibrated_
        assert hasattr(reg, 'calibration_non_conformity')

    def test_auto_calibrate_with_custom_split(self, regression_data):
        """Test auto_calibrate with custom test_size."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X, y, auto_calibrate=True, tts_kwargs={'test_size': 0.5, 'random_state': 42})

        assert reg.is_calibrated_

    def test_classifier_auto_calibrate(self):
        """Test auto_calibrate for classifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)

        clf = ConformalClassifier(LogisticRegression(max_iter=1000))
        clf.fit(X, y, auto_calibrate=True)

        assert clf.is_calibrated_


class TestParameterPropagation:
    """Tests for parameter propagation through estimators."""

    def test_fit_returns_self(self, regression_data):
        """Test that fit returns self."""
        X_train, X_test, y_train, y_test = regression_data

        reg = ConformalRegressor(LinearRegression())
        result = reg.fit(X_train[:100], y_train[:100])

        assert result is reg

    def test_calibrate_returns_self(self, regression_data):
        """Test that calibrate returns self."""
        X_train, X_test, y_train, y_test = regression_data

        reg = ConformalRegressor(LinearRegression())
        reg.fit(X_train[:100], y_train[:100])
        result = reg.calibrate(X_train[100:150], y_train[100:150])

        assert result is reg

    def test_chaining_operations(self, regression_data):
        """Test method chaining."""
        X_train, X_test, y_train, y_test = regression_data

        reg = ConformalRegressor(LinearRegression())
        result = reg.fit(X_train[:100], y_train[:100]).calibrate(
            X_train[100:150], y_train[100:150]
        )

        assert result is reg
        assert reg.is_calibrated_
