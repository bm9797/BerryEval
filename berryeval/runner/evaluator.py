"""Evaluation runner orchestration."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from berryeval import metrics
from berryeval.dataset.reader import read_dataset_metadata, read_dataset_records
from berryeval.metrics._types import PADDING_ID
from berryeval.persistence.types import (
    LatencyReport,
    MetricValue,
    QueryBreakdown,
    RunResult,
)
from berryeval.runner.latency import LatencyTracker

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from berryeval.config.eval_config import EvalConfig
    from berryeval.retrievers.base import RetrieverAdapter


class IDMapper:
    """Maps string document IDs to stable integer IDs for metric kernels."""

    def __init__(self) -> None:
        self._str_to_int: dict[str, int] = {}
        self._next_id = 0

    def get_or_assign(self, doc_id: str) -> int:
        if doc_id not in self._str_to_int:
            self._str_to_int[doc_id] = self._next_id
            self._next_id += 1
        return self._str_to_int[doc_id]

    def map_ids(self, doc_ids: list[str]) -> list[int]:
        return [self.get_or_assign(doc_id) for doc_id in doc_ids]


class EvaluationRunner:
    """Orchestrates dataset -> retriever -> metrics execution."""

    def __init__(self, adapter: RetrieverAdapter, config: EvalConfig) -> None:
        self._adapter = adapter
        self._config = config

    def run(
        self,
        dataset_path: Path,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> RunResult:
        metadata = read_dataset_metadata(dataset_path)
        records = list(read_dataset_records(dataset_path))

        if not records:
            msg = f"Dataset has no records: {dataset_path}"
            raise ValueError(msg)

        max_k = max(self._config.evaluation.k_values)
        mapper = IDMapper()
        latency_tracker = LatencyTracker()

        query_texts: list[str] = []
        retrieved_strings: list[list[str]] = []
        relevant_strings: list[list[str]] = []

        for i, record in enumerate(records):
            start = time.perf_counter()
            retrieved_docs = self._adapter.retrieve(record.query, top_k=max_k)
            elapsed = time.perf_counter() - start
            latency_tracker.record(elapsed)

            retrieved_ids = [doc.doc_id for doc in retrieved_docs][:max_k]
            relevant_ids = list(record.relevant_chunk_ids)

            mapper.map_ids(retrieved_ids)
            mapper.map_ids(relevant_ids)

            query_texts.append(record.query)
            retrieved_strings.append(retrieved_ids)
            relevant_strings.append(relevant_ids)

            if on_progress is not None:
                on_progress(i + 1, len(records))

        mapped_retrieved = [mapper.map_ids(ids) for ids in retrieved_strings]
        mapped_relevant = [mapper.map_ids(ids) for ids in relevant_strings]

        retrieved_array = self._build_padded_array(mapped_retrieved, width=max_k)
        max_relevant = max((len(ids) for ids in mapped_relevant), default=0)
        relevant_array = self._build_padded_array(mapped_relevant, width=max_relevant)

        metric_values: list[MetricValue] = []
        per_query_scores: list[dict[str, float]] = [{} for _ in range(len(records))]

        metric_fns = {
            "recall": metrics.recall_at_k,
            "precision": metrics.precision_at_k,
            "mrr": metrics.mrr,
            "ndcg": metrics.ndcg,
            "hit_rate": metrics.hit_rate,
        }

        for k in self._config.evaluation.k_values:
            for metric_name in self._config.evaluation.metrics:
                scores = metric_fns[metric_name](retrieved_array, relevant_array, k)
                mean_value = 0.0 if scores.size == 0 else float(np.mean(scores))
                values = (
                    [float(score) for score in scores]
                    if self._config.evaluation.per_query
                    else []
                )

                full_name = f"{metric_name}@{k}"
                metric_values.append(
                    MetricValue(
                        name=full_name,
                        k=k,
                        mean=mean_value,
                        values=values,
                    )
                )

                if self._config.evaluation.per_query:
                    for index, score in enumerate(scores):
                        per_query_scores[index][full_name] = float(score)

        latency_stats = latency_tracker.compute_stats()
        latency_report = LatencyReport(
            p50=latency_stats.p50 * 1000,
            p95=latency_stats.p95 * 1000,
            p99=latency_stats.p99 * 1000,
            mean=latency_stats.mean * 1000,
            min_ms=latency_stats.min * 1000,
            max_ms=latency_stats.max * 1000,
            count=latency_stats.count,
        )

        breakdowns: list[QueryBreakdown] = []
        if self._config.evaluation.per_query:
            latencies_ms = [
                duration * 1000 for duration in latency_tracker.get_latencies()
            ]
            for index, query in enumerate(query_texts):
                breakdowns.append(
                    QueryBreakdown(
                        query=query,
                        relevant_ids=relevant_strings[index],
                        retrieved_ids=retrieved_strings[index],
                        scores=per_query_scores[index],
                        latency_ms=latencies_ms[index],
                    )
                )

        return RunResult(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            config=self._config.model_dump(mode="python"),
            dataset_path=str(dataset_path),
            dataset_metadata=metadata.to_dict(),
            num_queries=len(records),
            metrics=metric_values,
            latency=latency_report,
            query_breakdowns=breakdowns,
        )

    @staticmethod
    def _build_padded_array(rows: list[list[int]], width: int) -> np.ndarray:
        arr = np.full((len(rows), width), PADDING_ID, dtype=np.int32)
        for i, row in enumerate(rows):
            if width == 0 or not row:
                continue
            clipped = row[:width]
            arr[i, : len(clipped)] = np.array(clipped, dtype=np.int32)
        return np.ascontiguousarray(arr, dtype=np.int32)
