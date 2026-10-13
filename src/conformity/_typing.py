"""Shared array contracts for dense, sparse, and DataFrame-compatible inputs.

ArrayLike includes objects implementing NumPy's array conversion protocol,
including pandas DataFrames. Labels can have any NumPy dtype; numeric predictions
and calibration scores use float64. Shape constraints are validated at runtime.
Estimator-specific metadata remains Any because its schema belongs to the wrapped
estimator, including sklearn pipelines and search objects.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import sparray, spmatrix
from typing_extensions import TypeAlias, TypedDict

FeatureMatrix: TypeAlias = Union[ArrayLike, spmatrix, sparray]
FloatArray: TypeAlias = NDArray[np.float64]


class TrainTestSplitKwargs(TypedDict, total=False):
    """Supported ``train_test_split`` options for automatic calibration."""

    test_size: Union[float, int, None]
    train_size: Union[float, int, None]
    random_state: Union[int, np.random.RandomState, None]
    shuffle: bool
    stratify: Optional[ArrayLike]
