"""Input validation for metric computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def validate_inputs(
    retrieved: NDArray[np.int32],
    relevant: NDArray[np.int32],
    k: int,
) -> None:
    """Validate metric inputs match the NumPy array contract.

    Args:
        retrieved: 2D int32 C-contiguous array of retrieved document IDs.
        relevant: 2D int32 C-contiguous array of relevant document IDs.
        k: Number of top results to consider.

    Raises:
        TypeError: If inputs are not numpy arrays.
        ValueError: If arrays have wrong dtype, shape, or layout.
    """
    for name, arr in [("retrieved", retrieved), ("relevant", relevant)]:
        if not isinstance(arr, np.ndarray):
            msg = f"{name} must be a numpy ndarray, got {type(arr).__name__}"
            raise TypeError(msg)
        if arr.dtype != np.int32:
            msg = f"{name} must have dtype int32, got {arr.dtype}"
            raise ValueError(msg)
        if arr.ndim != 2:
            msg = f"{name} must be 2D, got {arr.ndim}D"
            raise ValueError(msg)
        if not arr.flags["C_CONTIGUOUS"]:
            msg = f"{name} must be C-contiguous"
            raise ValueError(msg)

    if k < 1:
        msg = f"k must be >= 1, got {k}"
        raise ValueError(msg)
    if k > retrieved.shape[1]:
        msg = f"k ({k}) exceeds retrieved columns ({retrieved.shape[1]})"
        raise ValueError(msg)
    if retrieved.shape[0] != relevant.shape[0]:
        msg = (
            f"Query count mismatch: retrieved has {retrieved.shape[0]} queries, "
            f"relevant has {relevant.shape[0]}"
        )
        raise ValueError(msg)
