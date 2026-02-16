"""Tests for Mean Reciprocal Rank (MRR) metric."""

from __future__ import annotations

import numpy as np

from berryeval.metrics import mrr


def _arr(data: list[list[int]]) -> np.ndarray:
    return np.array(data, dtype=np.int32)


class TestMRR:
    """Test suite for MRR."""

    def test_first_position(self) -> None:
        """Relevant at rank 1 -> RR = 1.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = mrr(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_second_position(self) -> None:
        """Rank 2 -> RR = 0.5."""
        retrieved = _arr([[10, 1, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = mrr(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.5], atol=1e-6)

    def test_fifth_position(self) -> None:
        """Rank 5 -> RR = 0.2."""
        retrieved = _arr([[10, 20, 30, 40, 1]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = mrr(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.2], atol=1e-6)

    def test_not_found(self) -> None:
        """No relevant in top-k -> 0.0."""
        retrieved = _arr([[10, 20, 30, 40, 50]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = mrr(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_multi_query(self) -> None:
        """Average across queries."""
        retrieved = _arr(
            [
                [1, 2, 3, 4, 5],  # RR = 1.0
                [10, 1, 3, 4, 5],  # RR = 0.5
                [10, 20, 30, 40, 1],  # RR = 0.2
            ]
        )
        relevant = _arr(
            [
                [1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1],
            ]
        )
        result = mrr(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0, 0.5, 0.2], atol=1e-6)

    def test_multiple_relevant(self) -> None:
        """MRR uses FIRST relevant doc only."""
        retrieved = _arr([[10, 1, 2, 3, 4]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = mrr(retrieved, relevant, k=5)
        # First relevant is 1 at rank 2
        np.testing.assert_allclose(result, [0.5], atol=1e-6)

    def test_empty_relevant(self) -> None:
        """All padding -> 0.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[-1, -1, -1, -1, -1]])
        result = mrr(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_with_padding(self) -> None:
        """Padding in relevant is excluded."""
        retrieved = _arr([[10, 20, 1, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = mrr(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0 / 3.0], atol=1e-6)

    def test_determinism(self) -> None:
        """100 identical calls produce identical results."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[3, -1, -1, -1, -1]])
        results = [mrr(retrieved, relevant, k=5) for _ in range(100)]
        for r in results:
            np.testing.assert_array_equal(r, results[0])

    def test_no_mutation(self) -> None:
        """Input arrays unchanged after call."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, -1, -1, -1]])
        retrieved_copy = retrieved.copy()
        relevant_copy = relevant.copy()
        mrr(retrieved, relevant, k=5)
        np.testing.assert_array_equal(retrieved, retrieved_copy)
        np.testing.assert_array_equal(relevant, relevant_copy)

    def test_result_dtype(self) -> None:
        """Result should be float32."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = mrr(retrieved, relevant, k=5)
        assert result.dtype == np.float32
