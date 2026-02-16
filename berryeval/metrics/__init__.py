"""BerryEval metrics engine with Python/C backend dispatch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from berryeval.metrics._validation import validate_inputs

if TYPE_CHECKING:
    from berryeval.metrics._types import MetricResult, QueryDocArray, RelevanceArray

logger = logging.getLogger(__name__)

# Backend dispatch: try C extension, fall back to pure Python
try:
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        hit_rate as _hit_rate,
    )
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        mrr as _mrr,
    )
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        ndcg as _ndcg,
    )
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        precision_at_k as _precision_at_k,
    )
    from berryeval.metrics._native import (  # type: ignore[attr-defined]
        recall_at_k as _recall_at_k,
    )

    BACKEND = "native"
    logger.debug("Using native C backend for metrics")
except ImportError:
    from berryeval.metrics._python import hit_rate as _hit_rate
    from berryeval.metrics._python import mrr as _mrr
    from berryeval.metrics._python import ndcg as _ndcg
    from berryeval.metrics._python import precision_at_k as _precision_at_k
    from berryeval.metrics._python import recall_at_k as _recall_at_k

    BACKEND = "python"
    logger.debug("C extension not available, using pure Python backend")


def get_backend() -> str:
    """Return the active metrics backend name."""
    return BACKEND


def recall_at_k(
    retrieved: QueryDocArray, relevant: RelevanceArray, k: int
) -> MetricResult:
    """Compute recall@k for each query."""
    validate_inputs(retrieved, relevant, k)
    return _recall_at_k(retrieved, relevant, k)  # type: ignore[no-any-return]


def precision_at_k(
    retrieved: QueryDocArray, relevant: RelevanceArray, k: int
) -> MetricResult:
    """Compute precision@k for each query."""
    validate_inputs(retrieved, relevant, k)
    return _precision_at_k(retrieved, relevant, k)  # type: ignore[no-any-return]


def mrr(retrieved: QueryDocArray, relevant: RelevanceArray, k: int) -> MetricResult:
    """Compute Mean Reciprocal Rank for each query."""
    validate_inputs(retrieved, relevant, k)
    return _mrr(retrieved, relevant, k)  # type: ignore[no-any-return]


def ndcg(retrieved: QueryDocArray, relevant: RelevanceArray, k: int) -> MetricResult:
    """Compute normalized Discounted Cumulative Gain for each query."""
    validate_inputs(retrieved, relevant, k)
    return _ndcg(retrieved, relevant, k)  # type: ignore[no-any-return]


def hit_rate(
    retrieved: QueryDocArray, relevant: RelevanceArray, k: int
) -> MetricResult:
    """Compute hit rate for each query."""
    validate_inputs(retrieved, relevant, k)
    return _hit_rate(retrieved, relevant, k)  # type: ignore[no-any-return]


__all__ = [
    "BACKEND",
    "get_backend",
    "hit_rate",
    "mrr",
    "ndcg",
    "precision_at_k",
    "recall_at_k",
]
