"""Shared fixtures for benchmark data generation."""

import numpy as np
import pytest

from berryeval.metrics._types import PADDING_ID

SCALES: dict[str, int] = {
    "1K": 1_000,
    "10K": 10_000,
    "100K": 100_000,
}


def make_benchmark_data(
    n_queries: int,
    n_retrieved: int = 50,
    n_relevant: int = 10,
    relevant_density: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate benchmark data for metric evaluation.

    Returns (retrieved, relevant) as 2D int32 C-contiguous arrays.
    """
    rng = np.random.default_rng(seed)

    # Retrieved: random doc IDs from 1..100_000
    retrieved = rng.integers(1, 100_001, size=(n_queries, n_retrieved), dtype=np.int32)

    # Relevant: partially filled with overlap from retrieved
    relevant = np.full((n_queries, n_relevant), PADDING_ID, dtype=np.int32)
    n_overlap = int(n_relevant * relevant_density)

    for i in range(n_queries):
        # Pick some doc IDs from retrieved (ensures non-zero scores)
        overlap_ids = rng.choice(retrieved[i], size=n_overlap, replace=False)
        # Pick remaining from random pool
        n_random = n_relevant - n_overlap
        random_ids = rng.integers(1, 100_001, size=n_random, dtype=np.int32)
        relevant[i, :n_overlap] = overlap_ids
        relevant[i, n_overlap:] = random_ids

    # Ensure C-contiguous
    retrieved = np.ascontiguousarray(retrieved, dtype=np.int32)
    relevant = np.ascontiguousarray(relevant, dtype=np.int32)

    return retrieved, relevant


@pytest.fixture(
    params=list(SCALES.keys()),
    ids=list(SCALES.keys()),
    scope="session",
)
def benchmark_data(request: pytest.FixtureRequest) -> tuple[np.ndarray, np.ndarray]:
    """Parametrized benchmark data at various scales."""
    scale_name = request.param
    n_queries = SCALES[scale_name]
    return make_benchmark_data(n_queries)


@pytest.fixture(scope="session")
def data_100k() -> tuple[np.ndarray, np.ndarray]:
    """100K query benchmark data."""
    return make_benchmark_data(100_000)
