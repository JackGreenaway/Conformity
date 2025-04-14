import numpy as np
from numpy.typing import ArrayLike


def prediction_set_coverage(y_true: ArrayLike, prediction_set: ArrayLike) -> float:
    return np.mean((prediction_set == y_true.reshape(-1, 1)).any(axis=1))


def prediction_set_efficiency(prediction_set: ArrayLike) -> float:
    return np.mean(
        (np.sum(~np.isnan(prediction_set), axis=1) - 1) / (prediction_set.shape[1] - 1)
    )


def prediction_interval_coverage(
    y_true: ArrayLike, prediction_interval: ArrayLike
) -> float:
    return np.mean(
        (prediction_interval[:, 0] <= y_true) & (y_true <= prediction_interval[:, 1])
    )


def prediction_interval_efficiency(
    point_prediction: ArrayLike, prediction_interval: ArrayLike
) -> float:
    return np.mean(
        (prediction_interval[:, 1] - prediction_interval[:, 0])
        / (np.abs(point_prediction) + 1e-10)  # constant to avoid division by zero
    )
