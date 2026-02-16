"""Shared test fixtures for BerryEval tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_retrieved() -> np.ndarray:
    """5 queries, 10 retrieved document IDs each."""
    return np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            [1, 11, 2, 12, 3, 13, 4, 14, 5, 15],
        ],
        dtype=np.int32,
    )


@pytest.fixture
def sample_relevant() -> np.ndarray:
    """5 queries, up to 5 relevant document IDs (padded with -1)."""
    return np.array(
        [
            [1, 3, 5, -1, -1],
            [11, 13, 15, 17, 19],
            [1, 2, 3, 4, 5],
            [1, 3, 5, 7, 9],
            [1, 2, 3, -1, -1],
        ],
        dtype=np.int32,
    )
