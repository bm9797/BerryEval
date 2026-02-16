"""Tests for metric input validation."""

from __future__ import annotations

import numpy as np
import pytest

from berryeval.metrics._validation import validate_inputs


class TestValidateInputs:
    """Test suite for validate_inputs function."""

    def _make_valid(
        self, n_queries: int = 3, n_retrieved: int = 10, n_relevant: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Helper to create valid test arrays."""
        retrieved = np.zeros((n_queries, n_retrieved), dtype=np.int32)
        relevant = np.zeros((n_queries, n_relevant), dtype=np.int32)
        return retrieved, relevant

    def test_rejects_non_numpy_input(self) -> None:
        """Pass list, expect TypeError."""
        _, relevant = self._make_valid()
        with pytest.raises(TypeError, match="must be a numpy ndarray"):
            validate_inputs([[1, 2, 3]], relevant, k=1)  # type: ignore[arg-type]

    def test_rejects_non_numpy_relevant(self) -> None:
        """Pass list for relevant, expect TypeError."""
        retrieved, _ = self._make_valid()
        with pytest.raises(TypeError, match="must be a numpy ndarray"):
            validate_inputs(retrieved, [[1, 2, 3]], k=1)  # type: ignore[arg-type]

    def test_rejects_wrong_dtype(self) -> None:
        """Pass float64 array, expect ValueError."""
        retrieved = np.zeros((3, 10), dtype=np.float64)
        _, relevant = self._make_valid()
        with pytest.raises(ValueError, match="must have dtype int32"):
            validate_inputs(retrieved, relevant, k=1)  # type: ignore[arg-type]

    def test_rejects_wrong_dtype_relevant(self) -> None:
        """Pass float64 relevant array, expect ValueError."""
        retrieved, _ = self._make_valid()
        relevant = np.zeros((3, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="must have dtype int32"):
            validate_inputs(retrieved, relevant, k=1)  # type: ignore[arg-type]

    def test_rejects_1d_array(self) -> None:
        """Pass 1D array, expect ValueError."""
        retrieved = np.zeros(10, dtype=np.int32)
        _, relevant = self._make_valid()
        with pytest.raises(ValueError, match="must be 2D"):
            validate_inputs(retrieved, relevant, k=1)  # type: ignore[arg-type]

    def test_rejects_non_contiguous(self) -> None:
        """Pass Fortran-order array, expect ValueError."""
        retrieved = np.asfortranarray(np.zeros((3, 10), dtype=np.int32))
        _, relevant = self._make_valid()
        with pytest.raises(ValueError, match="must be C-contiguous"):
            validate_inputs(retrieved, relevant, k=1)

    def test_rejects_k_zero(self) -> None:
        """k=0, expect ValueError."""
        retrieved, relevant = self._make_valid()
        with pytest.raises(ValueError, match="k must be >= 1"):
            validate_inputs(retrieved, relevant, k=0)

    def test_rejects_k_exceeds_columns(self) -> None:
        """k > columns, expect ValueError."""
        retrieved, relevant = self._make_valid(n_retrieved=5)
        with pytest.raises(ValueError, match=r"k.*exceeds retrieved columns"):
            validate_inputs(retrieved, relevant, k=6)

    def test_rejects_query_count_mismatch(self) -> None:
        """Different row counts, expect ValueError."""
        retrieved = np.zeros((3, 10), dtype=np.int32)
        relevant = np.zeros((5, 5), dtype=np.int32)
        with pytest.raises(ValueError, match="Query count mismatch"):
            validate_inputs(retrieved, relevant, k=1)

    def test_accepts_valid_inputs(self) -> None:
        """Valid arrays pass without error."""
        retrieved, relevant = self._make_valid()
        validate_inputs(retrieved, relevant, k=5)  # Should not raise
