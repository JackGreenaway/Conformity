import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.typing import ArrayLike


def prediction_set_coverage(y_true: ArrayLike, prediction_set: ArrayLike) -> float:
    """
    Calculate the coverage of prediction sets in conformal prediction.

    Parameters
    ----------
    y_true : ArrayLike
        True target values.
    prediction_set : ArrayLike
        Array of prediction sets for each sample.

    Returns
    -------
    float
        Proportion of samples for which the true value is contained in the prediction set.
    """
    return np.mean((prediction_set == y_true.reshape(-1, 1)).any(axis=1))


def prediction_set_efficiency(prediction_set: ArrayLike) -> float:
    """
    Compute the efficiency of prediction sets in conformal prediction.

    Efficiency is measured as the average size of the prediction set, normalised by the maximum possible size.

    Parameters
    ----------
    prediction_set : ArrayLike
        Array of prediction sets for each sample.

    Returns
    -------
    float
        Average normalised size of the prediction sets.
    """
    return np.mean(
        (np.sum(~np.isnan(prediction_set), axis=1) - 1) / (prediction_set.shape[1] - 1)
    )


def prediction_interval_coverage(
    y_true: ArrayLike, prediction_intervals: ArrayLike
) -> float:
    """
    Calculate the coverage of prediction intervals in conformal prediction.

    Parameters
    ----------
    y_true : ArrayLike
        True target values.
    prediction_intervals : ArrayLike
        Array of prediction intervals, shape (n_samples, 2).

    Returns
    -------
    float
        Proportion of samples for which the true value falls within the prediction interval.
    """
    return np.mean(
        (prediction_intervals[:, 0] <= y_true) & (y_true <= prediction_intervals[:, 1])
    )


def prediction_interval_efficiency(
    point_prediction: ArrayLike, prediction_intervals: ArrayLike, relative: bool = False
) -> float:
    """
    Compute the efficiency of prediction intervals in conformal prediction.

    Efficiency is measured as the average width of the prediction interval, optionally normalised by the absolute value of the point prediction.

    Parameters
    ----------
    point_prediction : ArrayLike
        Point predictions for each sample.
    prediction_intervals : ArrayLike
        Array of prediction intervals, shape (n_samples, 2).
    relative : bool, default=False
        If True, normalise interval width by the absolute value of the point prediction.

    Returns
    -------
    float
        Average (normalised) width of the prediction intervals.
    """
    relative = (
        np.abs(point_prediction) + 1e-10  # constant to avoid division by zero
        if relative
        else 1
    )

    return np.mean((prediction_intervals[:, 1] - prediction_intervals[:, 0]) / relative)


def prediction_interval_ratio(
    point_predictions: ArrayLike,
    prediction_intervals: ArrayLike,
) -> float:
    """
    Calculate the mean ratio of the upper bound of prediction intervals to point predictions.

    Parameters
    ----------
    point_predictions : ArrayLike
        Point predictions for each sample.
    prediction_intervals : ArrayLike
        Array of prediction intervals, shape (n_samples, 2).

    Returns
    -------
    float
        Mean ratio of upper interval bound to point prediction.
    """
    return np.mean(prediction_intervals[:, 1] / point_predictions)


def prediction_interval_mse(
    y_true: ArrayLike, prediction_intervals: ArrayLike
) -> tuple[float, float]:
    """
    Compute the mean squared error (MSE) between the true values and the bounds of prediction intervals.

    Parameters
    ----------
    y_true : ArrayLike
        True target values.
    prediction_intervals : ArrayLike
        Array of prediction intervals, shape (n_samples, 2).

    Returns
    -------
    tuple of float
        MSE for the lower bound and MSE for the upper bound of the prediction intervals.
    """
    return tuple(
        [
            mean_squared_error(y_true=y_true, y_pred=prediction_intervals[:, i])
            for i in range(prediction_intervals.shape[1])
        ]
    )
