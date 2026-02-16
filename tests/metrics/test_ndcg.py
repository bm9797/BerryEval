"""Tests for normalized Discounted Cumulative Gain (nDCG) metric."""

from __future__ import annotations

import numpy as np

from berryeval.metrics import ndcg


def _arr(data: list[list[int]]) -> np.ndarray:
    return np.array(data, dtype=np.int32)


class TestNDCG:
    """Test suite for nDCG."""

    def test_perfect_ranking(self) -> None:
        """All relevant at top -> 1.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0], atol=1e-4)

    def test_worst_ranking(self) -> None:
        """Relevant at last position -> < 1.0."""
        retrieved = _arr([[10, 20, 30, 40, 1]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        # DCG = 1/log2(6), IDCG = 1/log2(2) = 1.0
        expected = (1.0 / np.log2(6)) / (1.0 / np.log2(2))
        np.testing.assert_allclose(result, [expected], atol=1e-4)
        assert result[0] < 1.0

    def test_no_relevant_docs(self) -> None:
        """Empty relevant -> 0.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[-1, -1, -1, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_single_relevant_at_1(self) -> None:
        """Single relevant at position 1 -> 1.0."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0], atol=1e-4)

    def test_single_relevant_at_k(self) -> None:
        """Single relevant at position k=3."""
        retrieved = _arr([[10, 20, 1, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        # DCG = 1/log2(4), IDCG = 1/log2(2) = 1.0
        expected = (1.0 / np.log2(4)) / 1.0
        np.testing.assert_allclose(result, [expected], atol=1e-4)

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
        result = ndcg(retrieved, relevant, k=5)
        assert result.shape == (2,)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-4)
        np.testing.assert_allclose(result[1], 1.0, atol=1e-4)

    def test_binary_relevance_values(self) -> None:
        """nDCG uses binary relevance (0 or 1)."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[2, 4, -1, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        # DCG = 0/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5) + 0/log2(6)
        dcg = 1.0 / np.log2(3) + 1.0 / np.log2(5)
        idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3)
        expected = dcg / idcg
        np.testing.assert_allclose(result, [expected], atol=1e-4)

    def test_idcg_computation(self) -> None:
        """Reference: k=5, relevant={1,2,3}, retrieved=[1,4,2,5,3]."""
        retrieved = _arr([[1, 4, 2, 5, 3]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6)
        dcg = 1.0 / np.log2(2) + 1.0 / np.log2(4) + 1.0 / np.log2(6)
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3) + 1.0 / np.log2(4)
        expected = dcg / idcg
        np.testing.assert_allclose(result, [expected], atol=1e-4)
        # Approximate value from spec
        np.testing.assert_allclose(result, [0.8856], atol=1e-3)

    def test_with_padding(self) -> None:
        """Padding in relevant is excluded."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, -1, -1, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        np.testing.assert_allclose(result, [1.0], atol=1e-4)

    def test_determinism(self) -> None:
        """100 identical calls produce identical results."""
        retrieved = _arr([[1, 4, 2, 5, 3]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        results = [ndcg(retrieved, relevant, k=5) for _ in range(100)]
        for r in results:
            np.testing.assert_array_equal(r, results[0])

    def test_no_mutation(self) -> None:
        """Input arrays unchanged after call."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 3, -1, -1, -1]])
        retrieved_copy = retrieved.copy()
        relevant_copy = relevant.copy()
        ndcg(retrieved, relevant, k=5)
        np.testing.assert_array_equal(retrieved, retrieved_copy)
        np.testing.assert_array_equal(relevant, relevant_copy)

    def test_result_dtype(self) -> None:
        """Result should be float32."""
        retrieved = _arr([[1, 2, 3, 4, 5]])
        relevant = _arr([[1, 2, 3, -1, -1]])
        result = ndcg(retrieved, relevant, k=5)
        assert result.dtype == np.float32
