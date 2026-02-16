"""Tests for persistence save/load."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from berryeval.persistence.results import load_result, save_result
from berryeval.persistence.types import (
    LatencyReport,
    MetricValue,
    QueryBreakdown,
    RunResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_run_result() -> RunResult:
    return RunResult(
        run_id="12345678-abcd-efgh-ijkl-1234567890ab",
        timestamp="2026-01-01T12:00:00+00:00",
        config={
            "retriever": {"type": "mock"},
            "evaluation": {"k_values": [5], "metrics": ["recall"]},
        },
        dataset_path="/tmp/dataset.jsonl",
        dataset_metadata={"model": "gpt-4", "version": "0.1.0"},
        num_queries=2,
        metrics=[
            MetricValue(name="recall@5", k=5, mean=0.75, values=[1.0, 0.5]),
            MetricValue(name="precision@5", k=5, mean=0.3, values=[0.4, 0.2]),
        ],
        latency=LatencyReport(
            p50=10.0,
            p95=20.0,
            p99=25.0,
            mean=12.5,
            min_ms=5.0,
            max_ms=30.0,
            count=2,
        ),
        query_breakdowns=[
            QueryBreakdown(
                query="q1",
                relevant_ids=["doc1"],
                retrieved_ids=["doc1", "doc2"],
                scores={"recall@5": 1.0},
                latency_ms=9.5,
            )
        ],
    )


class TestSaveResult:
    def test_save_creates_file(self, tmp_path: Path):
        result = _make_run_result()

        saved_path = save_result(result, tmp_path)

        assert saved_path.exists()
        assert saved_path.name.startswith("berryeval_run_20260101_120000_12345678")

        loaded_json = json.loads(saved_path.read_text(encoding="utf-8"))
        assert loaded_json["run_id"] == result.run_id
        assert loaded_json["metrics"][0]["name"] == "recall@5"


class TestLoadResult:
    def test_load_existing_json(self, tmp_path: Path):
        filepath = tmp_path / "result.json"
        data = _make_run_result().model_dump(mode="python")
        filepath.write_text(json.dumps(data), encoding="utf-8")

        loaded = load_result(filepath)

        assert loaded.run_id == data["run_id"]
        assert loaded.num_queries == 2
        assert loaded.metrics[1].name == "precision@5"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_result(tmp_path / "missing.json")


class TestRoundTrip:
    def test_save_and_load_round_trip(self, tmp_path: Path):
        original = _make_run_result()

        filepath = save_result(original, tmp_path)
        loaded = load_result(filepath)

        assert loaded.model_dump(mode="python") == original.model_dump(mode="python")
