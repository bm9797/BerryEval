"""Tests for evaluation runner orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from berryeval.config.eval_config import EvalConfig
from berryeval.config.types import DatasetMetadata, DatasetRecord
from berryeval.dataset.writer import write_dataset
from berryeval.retrievers.base import RetrievedDocument, RetrieverAdapter
from berryeval.runner.evaluator import EvaluationRunner

if TYPE_CHECKING:
    from pathlib import Path


class MockRetriever(RetrieverAdapter):
    name = "mock"

    def __init__(self, results_map: dict[str, list[RetrievedDocument]]) -> None:
        self._results_map = results_map

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        return self._results_map.get(query, [])[:top_k]

    def close(self) -> None:
        return


def _make_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        config_hash="test-hash",
        model="gpt-4",
        chunk_size=800,
        overlap=100,
        timestamp="2026-01-01T00:00:00+00:00",
        corpus_stats={"files": 2, "total_chunks": 2},
        version="0.1.0",
        prompt_template="test: {chunk}",
    )


def _write_dataset(filepath: Path, records: list[DatasetRecord]) -> None:
    write_dataset(filepath, _make_metadata(), records)


def _make_config(
    *,
    k_values: list[int],
    metrics: list[str],
    per_query: bool = False,
) -> EvalConfig:
    return EvalConfig(
        retriever={"type": "mock"},
        evaluation={
            "k_values": k_values,
            "metrics": metrics,
            "per_query": per_query,
        },
    )


def _get_metric(result_name: str, result_metrics: list) -> float:
    for metric in result_metrics:
        if metric.name == result_name:
            return metric.mean
    msg = f"Metric not found: {result_name}"
    raise AssertionError(msg)


class TestEvaluationRunner:
    def test_run_basic(self, tmp_path: Path):
        records = [
            DatasetRecord(
                query="q1",
                relevant_chunk_ids=["doc1", "doc2"],
                chunk_text="a",
            ),
            DatasetRecord(
                query="q2",
                relevant_chunk_ids=["doc3"],
                chunk_text="b",
            ),
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever(
            {
                "q1": [
                    RetrievedDocument(doc_id="doc1", score=0.9),
                    RetrievedDocument(doc_id="doc2", score=0.8),
                ],
                "q2": [
                    RetrievedDocument(doc_id="doc3", score=0.7),
                    RetrievedDocument(doc_id="doc4", score=0.6),
                ],
            }
        )
        config = _make_config(
            k_values=[2],
            metrics=["recall", "precision", "mrr", "ndcg", "hit_rate"],
        )

        runner = EvaluationRunner(retriever, config)
        result = runner.run(dataset_path)

        assert result.num_queries == 2
        assert len(result.metrics) == 5
        assert _get_metric("recall@2", result.metrics) == pytest.approx(1.0)
        assert _get_metric("precision@2", result.metrics) == pytest.approx(0.75)
        assert _get_metric("mrr@2", result.metrics) == pytest.approx(1.0)
        assert _get_metric("ndcg@2", result.metrics) == pytest.approx(1.0)
        assert _get_metric("hit_rate@2", result.metrics) == pytest.approx(1.0)
        assert result.latency.count == 2

    def test_run_partial_retrieval(self, tmp_path: Path):
        records = [
            DatasetRecord(
                query="q1",
                relevant_chunk_ids=["doc1", "doc2"],
                chunk_text="a",
            ),
            DatasetRecord(query="q2", relevant_chunk_ids=["doc3"], chunk_text="b"),
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever(
            {
                "q1": [
                    RetrievedDocument(doc_id="doc1", score=1.0),
                    RetrievedDocument(doc_id="docx", score=0.5),
                ],
                "q2": [
                    RetrievedDocument(doc_id="docy", score=0.4),
                    RetrievedDocument(doc_id="docz", score=0.3),
                ],
            }
        )
        config = _make_config(k_values=[2], metrics=["recall", "precision"])

        result = EvaluationRunner(retriever, config).run(dataset_path)
        assert _get_metric("recall@2", result.metrics) == pytest.approx(0.25)
        assert _get_metric("precision@2", result.metrics) == pytest.approx(0.25)

    def test_run_multiple_k_values(self, tmp_path: Path):
        records = [
            DatasetRecord(
                query="q1",
                relevant_chunk_ids=["doc1", "doc2"],
                chunk_text="a",
            )
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever(
            {
                "q1": [
                    RetrievedDocument(doc_id="doc1", score=0.9),
                    RetrievedDocument(doc_id="doc2", score=0.8),
                    RetrievedDocument(doc_id="doc3", score=0.7),
                    RetrievedDocument(doc_id="doc4", score=0.6),
                    RetrievedDocument(doc_id="doc5", score=0.5),
                ]
            }
        )
        config = _make_config(
            k_values=[1, 3, 5],
            metrics=["recall", "precision", "mrr", "ndcg", "hit_rate"],
        )

        result = EvaluationRunner(retriever, config).run(dataset_path)
        assert len(result.metrics) == 15

    def test_run_per_query_breakdowns(self, tmp_path: Path):
        records = [
            DatasetRecord(query="q1", relevant_chunk_ids=["doc1"], chunk_text="a"),
            DatasetRecord(query="q2", relevant_chunk_ids=["doc2"], chunk_text="b"),
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever(
            {
                "q1": [RetrievedDocument(doc_id="doc1", score=1.0)],
                "q2": [RetrievedDocument(doc_id="doc2", score=1.0)],
            }
        )
        config = _make_config(
            k_values=[1],
            metrics=["recall", "precision", "mrr"],
            per_query=True,
        )

        result = EvaluationRunner(retriever, config).run(dataset_path)
        assert len(result.query_breakdowns) == 2
        assert result.query_breakdowns[0].query == "q1"
        assert "recall@1" in result.query_breakdowns[0].scores

    def test_run_no_per_query(self, tmp_path: Path):
        records = [
            DatasetRecord(query="q1", relevant_chunk_ids=["doc1"], chunk_text="a")
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever({"q1": [RetrievedDocument(doc_id="doc1", score=1.0)]})
        config = _make_config(k_values=[1], metrics=["recall"], per_query=False)

        result = EvaluationRunner(retriever, config).run(dataset_path)
        assert result.query_breakdowns == []
        assert result.metrics[0].values == []

    def test_run_empty_retrieval(self, tmp_path: Path):
        records = [
            DatasetRecord(query="q1", relevant_chunk_ids=["doc1"], chunk_text="a")
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever({})
        config = _make_config(
            k_values=[2],
            metrics=["recall", "precision", "mrr", "ndcg", "hit_rate"],
        )

        result = EvaluationRunner(retriever, config).run(dataset_path)
        for metric in result.metrics:
            assert metric.mean == pytest.approx(0.0)

    def test_run_subset_metrics(self, tmp_path: Path):
        records = [
            DatasetRecord(query="q1", relevant_chunk_ids=["doc1"], chunk_text="a")
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever({"q1": [RetrievedDocument(doc_id="doc1", score=1.0)]})
        config = _make_config(k_values=[1], metrics=["recall", "mrr"])

        result = EvaluationRunner(retriever, config).run(dataset_path)
        names = [metric.name for metric in result.metrics]
        assert names == ["recall@1", "mrr@1"]

    def test_id_mapping_consistency(self, tmp_path: Path):
        records = [
            DatasetRecord(query="q1", relevant_chunk_ids=["doc1"], chunk_text="a"),
            DatasetRecord(query="q2", relevant_chunk_ids=["doc1"], chunk_text="b"),
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever(
            {
                "q1": [RetrievedDocument(doc_id="doc1", score=1.0)],
                "q2": [RetrievedDocument(doc_id="doc1", score=1.0)],
            }
        )
        config = _make_config(k_values=[1], metrics=["recall"])

        result = EvaluationRunner(retriever, config).run(dataset_path)
        assert _get_metric("recall@1", result.metrics) == pytest.approx(1.0)

    def test_progress_callback(self, tmp_path: Path):
        records = [
            DatasetRecord(query="q1", relevant_chunk_ids=["doc1"], chunk_text="a"),
            DatasetRecord(query="q2", relevant_chunk_ids=["doc2"], chunk_text="b"),
            DatasetRecord(query="q3", relevant_chunk_ids=["doc3"], chunk_text="c"),
        ]
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, records)

        retriever = MockRetriever(
            {
                "q1": [RetrievedDocument(doc_id="doc1", score=1.0)],
                "q2": [RetrievedDocument(doc_id="doc2", score=1.0)],
                "q3": [RetrievedDocument(doc_id="doc3", score=1.0)],
            }
        )
        config = _make_config(k_values=[1], metrics=["recall"])

        calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            calls.append((current, total))

        result = EvaluationRunner(retriever, config).run(dataset_path, on_progress)
        assert result.num_queries == 3
        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_empty_dataset_raises(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        _write_dataset(dataset_path, [])

        retriever = MockRetriever({})
        config = _make_config(k_values=[1], metrics=["recall"])

        with pytest.raises(ValueError, match="no records"):
            EvaluationRunner(retriever, config).run(dataset_path)
