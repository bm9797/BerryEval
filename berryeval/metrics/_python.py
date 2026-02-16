"""Pure Python implementations of IR metrics.

These serve as the always-available fallback and the reference specification
that future C kernels must match exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from berryeval.metrics._types import PADDING_ID

if TYPE_CHECKING:
    from berryeval.metrics._types import MetricResult, QueryDocArray, RelevanceArray


def recall_at_k(
    retrieved: QueryDocArray, relevant: RelevanceArray, k: int
) -> MetricResult:
    """Compute recall@k for each query."""
    n_queries = retrieved.shape[0]
    scores = np.zeros(n_queries, dtype=np.float64)

    for i in range(n_queries):
        rel_set = set(relevant[i][relevant[i] != PADDING_ID].tolist())
        if len(rel_set) == 0:
            scores[i] = 0.0
            continue
        ret_set = set(retrieved[i, :k].tolist())
        scores[i] = len(ret_set & rel_set) / len(rel_set)

    return scores.astype(np.float32)


def precision_at_k(
    retrieved: QueryDocArray, relevant: RelevanceArray, k: int
) -> MetricResult:
    """Compute precision@k for each query."""
    n_queries = retrieved.shape[0]
    scores = np.zeros(n_queries, dtype=np.float64)

    for i in range(n_queries):
        rel_set = set(relevant[i][relevant[i] != PADDING_ID].tolist())
        ret_set = set(retrieved[i, :k].tolist())
        scores[i] = len(ret_set & rel_set) / k

    return scores.astype(np.float32)


def mrr(retrieved: QueryDocArray, relevant: RelevanceArray, k: int) -> MetricResult:
    """Compute Mean Reciprocal Rank for each query."""
    n_queries = retrieved.shape[0]
    scores = np.zeros(n_queries, dtype=np.float64)

    for i in range(n_queries):
        rel_set = set(relevant[i][relevant[i] != PADDING_ID].tolist())
        if len(rel_set) == 0:
            scores[i] = 0.0
            continue
        for rank, doc_id in enumerate(retrieved[i, :k], start=1):
            if int(doc_id) in rel_set:
                scores[i] = 1.0 / rank
                break

    return scores.astype(np.float32)


def ndcg(retrieved: QueryDocArray, relevant: RelevanceArray, k: int) -> MetricResult:
    """Compute normalized Discounted Cumulative Gain for each query."""
    n_queries = retrieved.shape[0]
    scores = np.zeros(n_queries, dtype=np.float64)

    for i in range(n_queries):
        rel_set = set(relevant[i][relevant[i] != PADDING_ID].tolist())
        n_relevant = len(rel_set)
        if n_relevant == 0:
            scores[i] = 0.0
            continue

        # DCG@k
        dcg = 0.0
        for j in range(k):
            if int(retrieved[i, j]) in rel_set:
                dcg += 1.0 / np.log2(j + 2)

        # IDCG@k
        idcg = 0.0
        for j in range(min(n_relevant, k)):
            idcg += 1.0 / np.log2(j + 2)

        scores[i] = dcg / idcg if idcg > 0.0 else 0.0

    return scores.astype(np.float32)


def hit_rate(
    retrieved: QueryDocArray, relevant: RelevanceArray, k: int
) -> MetricResult:
    """Compute hit rate for each query."""
    n_queries = retrieved.shape[0]
    scores = np.zeros(n_queries, dtype=np.float64)

    for i in range(n_queries):
        rel_set = set(relevant[i][relevant[i] != PADDING_ID].tolist())
        if len(rel_set) == 0:
            scores[i] = 0.0
            continue
        ret_set = set(retrieved[i, :k].tolist())
        scores[i] = 1.0 if len(ret_set & rel_set) > 0 else 0.0

    return scores.astype(np.float32)
