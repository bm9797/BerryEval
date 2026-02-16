"""Tests for precision@k metric."""

from __future__ import annotations

import numpy as np

from berryeval.metrics import precision_at_k


def _arr(data: list[list[int]]) -> np.ndarray:
    return np.array(data, dtype=np.int32)


class TestPrecisionAtK:
    """Test suite for precision_at_k."""

    def test_perfect_precision(self) -> None:
        """All top-k relevant -> 1.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 2, 3, 4, 5]])
        result = precision_at_k(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_zero_precision(self) -> None:
        """None relevant -> 0.0."""
        retrieved = _arr([[10, 20, 30, 40, 50]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = precision_at_k(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_partial_precision(self) -> None:
        """2 relevant in top-5 -> 0.4."""
        retrieved = _arr([[1, 10, 2, 20, 30]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = precision_at_k(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.4], atol=1e-6)

    def test_precision_at_1(self) -> None:
        """Single position, hit."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = precision_at_k(retrieved, relevant, k=1)
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_precision_at_1_miss(self) -> None:
        """Single position, miss."""
        retrieved = _arr([[2, 1, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = precision_at_k(retrieved, relevant, k=1)
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
                [10, -1, -1, -1, -1],
            ]
        )
        result = precision_at_k(retrieved, relevant, k=5)
        # Query 1: 3/5=0.6, Query 2: 1/5=0.2
        np.testing.assert_allclose(result, [0.6, 0.2], atol=1e-6)

    def test_with_padding(self) -> None:
        """Padding in retrieved is ignored for intersection (only relevant padding matters)."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, -1, -1, -1]])
        result = precision_at_k(retrieved, relevant, k=5)
        # 2 relevant found / 5 = 0.4
        np.testing.assert_allclose(result, [0.4], atol=1e-6)

    def test_determinism(self) -> None:
        """100 identical calls produce identical results."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, 5, -1, -1]])
        results = [precision_at_k(retrieved, relevant, k=5) for _ in range(100)]
        for r in results:
            np.testing.assert_array_equal(r, results[0])

    def test_no_mutation(self) -> None:
        """Input arrays unchanged after call."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, -1, -1, -1]])
        retrieved_copy = retrieved.copy()
        relevant_copy = relevant.copy()
        precision_at_k(retrieved, relevant, k=5)
        np.testing.assert_array_equal(retrieved, retrieved_copy)
        np.testing.assert_array_equal(relevant, relevant_copy)

    def test_result_dtype(self) -> None:
        """Result should be float32."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = precision_at_k(retrieved, relevant, k=5)
        assert result.dtype == np.float32
