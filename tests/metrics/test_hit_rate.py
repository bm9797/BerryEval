"""Tests for hit rate metric."""

from __future__ import annotations

import numpy as np

from berryeval.metrics import hit_rate


def _arr(data: list[list[int]]) -> np.ndarray:
    return np.array(data, dtype=np.int32)


class TestHitRate:
    """Test suite for hit_rate."""

    def test_hit(self) -> None:
        """At least one relevant in top-k -> 1.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[3, -1, -1, -1, -1]])
        result = hit_rate(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_miss(self) -> None:
        """None relevant -> 0.0."""
        retrieved = _arr([[10, 20, 30, 40, 50]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = hit_rate(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_hit_at_boundary(self) -> None:
        """Relevant at exactly position k -> 1.0."""
        retrieved = _arr([[10, 20, 30, 40, 1]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = hit_rate(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_multi_query(self) -> None:
        """Mix of hits and misses."""
        retrieved = _arr(
            [
                [1, 2, 3, 4, 5],
                [10, 20, 30, 40, 50],
                [1, 10, 20, 30, 40],
            ]
        )
        relevant = _arr(
            [
                [1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1],
            ]
        )
        result = hit_rate(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0, 0.0, 1.0], atol=1e-6)

    def test_empty_relevant(self) -> None:
        """All padding -> 0.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[-1, -1, -1, -1, -1]])
        result = hit_rate(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_with_padding(self) -> None:
        """Padding in relevant is excluded."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[3, 5, -1, -1, -1]])
        result = hit_rate(retrieved, relevant, k=3)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_determinism(self) -> None:
        """100 identical calls produce identical results."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[3, -1, -1, -1, -1]])
        results = [hit_rate(retrieved, relevant, k=5) for _ in range(100)]
        for r in results:
            np.testing.assert_array_equal(r, results[0])

    def test_no_mutation(self) -> None:
        """Input arrays unchanged after call."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, -1, -1, -1]])
        retrieved_copy = retrieved.copy()
        relevant_copy = relevant.copy()
        hit_rate(retrieved, relevant, k=5)
        np.testing.assert_array_equal(retrieved, retrieved_copy)
        np.testing.assert_array_equal(relevant, relevant_copy)

    def test_result_dtype(self) -> None:
        """Result should be float32."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = hit_rate(retrieved, relevant, k=5)
        assert result.dtype == np.float32
