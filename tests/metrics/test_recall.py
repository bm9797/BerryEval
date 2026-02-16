"""Tests for recall@k metric."""

from __future__ import annotations

import numpy as np

from berryeval.metrics import recall_at_k


def _arr(data: list[list[int]]) -> np.ndarray:
    return np.array(data, dtype=np.int32)


class TestRecallAtK:
    """Test suite for recall_at_k."""

    def test_perfect_recall(self) -> None:
        """All relevant docs retrieved -> 1.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = recall_at_k(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_zero_recall(self) -> None:
        """No relevant docs retrieved -> 0.0."""
        retrieved = _arr([[10, 20, 30, 40, 50]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = recall_at_k(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_partial_recall(self) -> None:
        """2 of 3 relevant -> 2/3."""
        retrieved = _arr([[1, 2, 10, 20, 30]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = recall_at_k(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [2.0 / 3.0], atol=1e-6)

    def test_recall_at_1(self) -> None:
        """Single position."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = recall_at_k(retrieved, relevant, k=1)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_recall_at_1_miss(self) -> None:
        """Single position, no hit."""
        retrieved = _arr([[2, 1, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = recall_at_k(retrieved, relevant, k=1)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_multi_query(self) -> None:
        """Multiple queries."""
        retrieved = _arr(
            [
                [1, 2, 3, 4, 5],
                [10, 20, 30, 40, 50],
            ]
        )
        relevant = _arr(
            [
                [1, 2, 3, -1, -1],
                [10, 20, -1, -1, -1],
            ]
        )
        result = recall_at_k(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-6)

    def test_empty_relevant(self) -> None:
        """All padding in relevant -> 0.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[-1, -1, -1, -1, -1]])
        result = recall_at_k(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_with_padding(self) -> None:
        """Variable-length with -1 padding."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, -1, -1, -1]])
        result = recall_at_k(retrieved, relevant, k=3)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_determinism(self) -> None:
        """100 identical calls produce identical results."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, 5, -1, -1]])
        results = [recall_at_k(retrieved, relevant, k=5) for _ in range(100)]
        for r in results:
            np.testing.assert_array_equal(r, results[0])

    def test_no_mutation(self) -> None:
        """Input arrays unchanged after call."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, -1, -1, -1]])
        retrieved_copy = retrieved.copy()
        relevant_copy = relevant.copy()
        recall_at_k(retrieved, relevant, k=5)
        np.testing.assert_array_equal(retrieved, retrieved_copy)
        np.testing.assert_array_equal(relevant, relevant_copy)

    def test_result_dtype(self) -> None:
        """Result should be float32."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = recall_at_k(retrieved, relevant, k=5)
        assert result.dtype == np.float32
