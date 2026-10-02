"""Exact finite-population rank checks, independent of Monte Carlo tolerance."""

import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from conformity import ConformalRegressor
from test_safety_and_api import classifier


@pytest.mark.parametrize("alpha", [0.01, 0.2, 0.35, 0.5, 0.8])
@pytest.mark.parametrize("scores", [[1, 2, 3, 4, 5], [1, 1, 2, 2, 3]])
def test_all_possible_future_ranks_obey_coverage_bound(alpha, scores):
    # Conditional on this unordered population, each held-out index is equally
    # likely to be the future observation. Calibration order is irrelevant.
    scores = np.array(scores)
    hits = []
    for future in range(len(scores)):
        model = ConformalRegressor(DummyRegressor(strategy="constant", constant=0))
        model.fit([[0], [1]], [0, 0])
        calibration = np.delete(scores, future)
        model.calibrate(np.zeros((len(calibration), 1)), calibration)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lower, upper = model.predict_interval([[0]], alpha=alpha)[0]
        hits.append(lower <= scores[future] <= upper)
    rank = int(np.ceil(len(scores) * (1 - alpha)))
    assert sum(hits) >= rank
    assert np.mean(hits) >= 1 - alpha
    if len(np.unique(scores)) == len(scores):
        assert sum(hits) == rank


@pytest.mark.parametrize("method", ["lac", "aps"])
@pytest.mark.parametrize("alpha", [0.01, 0.2, 0.35, 0.5, 0.8])
def test_classification_rank_coverage_and_p_value_inversion(method, alpha):
    X = np.array([[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.5, 0.5], [0.2, 0.8]])
    y = np.array(["cat", "dog", "cat", "dog", "dog"])
    hits = []
    p_values = []
    for future in range(len(y)):
        model = classifier(method=method)
        keep = np.arange(len(y)) != future
        model.calibrate(X[keep], y[keep])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mask = model.predict_set(X[future : future + 1], alpha=alpha)
        p = model.predict_p_values(X[future : future + 1])
        np.testing.assert_array_equal(mask, p > alpha)
        true_column = np.flatnonzero(model.classes_ == y[future])[0]
        hits.append(mask[0, true_column])
        p_values.append(p[0, true_column])
    assert np.mean(hits) >= 1 - alpha
    assert np.mean(np.array(p_values) <= alpha) <= alpha
