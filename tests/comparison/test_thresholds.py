"""Tests for threshold parsing and checking."""

from __future__ import annotations

import pytest

from berryeval.comparison.thresholds import check_thresholds, parse_thresholds
from berryeval.persistence.types import LatencyReport, MetricValue, RunResult


def _make_latency() -> LatencyReport:
    return LatencyReport(
        p50=10.0, p95=20.0, p99=30.0, mean=12.0, min_ms=5.0, max_ms=50.0, count=10
    )


def _make_run_result(
    metrics: list[MetricValue] | None = None,
) -> RunResult:
    return RunResult(
        run_id="run-1",
        timestamp="2026-01-01T00:00:00Z",
        config={"model": "test"},
        dataset_path="data.jsonl",
        dataset_metadata={},
        num_queries=10,
        metrics=metrics or [],
        latency=_make_latency(),
    )


class TestParseThresholds:
    def test_single_threshold(self):
        result = parse_thresholds("recall@10=0.80")
        assert result == {"recall@10": 0.80}

    def test_multiple_thresholds(self):
        result = parse_thresholds("recall@10=0.80,precision@5=0.60")
        assert result == {"recall@10": 0.80, "precision@5": 0.60}

    def test_whitespace_handling(self):
        result = parse_thresholds(" recall@10 = 0.80 , precision@5 = 0.60 ")
        assert result == {"recall@10": 0.80, "precision@5": 0.60}

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_thresholds("")

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="missing '='"):
            parse_thresholds("recall@10:0.80")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="Non-numeric"):
            parse_thresholds("recall@10=abc")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_thresholds("=0.80")

    def test_integer_value(self):
        result = parse_thresholds("recall@10=1")
        assert result == {"recall@10": 1.0}


class TestCheckThresholds:
    def test_all_pass(self):
        run = _make_run_result(
            metrics=[
                MetricValue(name="recall@10", k=10, mean=0.90, values=[]),
                MetricValue(name="precision@5", k=5, mean=0.70, values=[]),
            ]
        )
        results = check_thresholds(run, {"recall@10": 0.80, "precision@5": 0.60})

        assert all(r.passed for r in results)

    def test_one_fails(self):
        run = _make_run_result(
            metrics=[
                MetricValue(name="recall@10", k=10, mean=0.70, values=[]),
                MetricValue(name="precision@5", k=5, mean=0.70, values=[]),
            ]
        )
        results = check_thresholds(run, {"recall@10": 0.80, "precision@5": 0.60})

        result_map = {r.metric_name: r for r in results}
        assert not result_map["recall@10"].passed
        assert result_map["precision@5"].passed

    def test_exact_threshold_passes(self):
        run = _make_run_result(
            metrics=[MetricValue(name="recall@10", k=10, mean=0.80, values=[])]
        )
        results = check_thresholds(run, {"recall@10": 0.80})

        assert results[0].passed is True

    def test_mixed_results(self):
        run = _make_run_result(
            metrics=[
                MetricValue(name="recall@10", k=10, mean=0.90, values=[]),
                MetricValue(name="precision@5", k=5, mean=0.50, values=[]),
            ]
        )
        results = check_thresholds(run, {"recall@10": 0.80, "precision@5": 0.60})

        result_map = {r.metric_name: r for r in results}
        assert result_map["recall@10"].passed
        assert not result_map["precision@5"].passed

    def test_unknown_metric(self):
        run = _make_run_result(
            metrics=[MetricValue(name="recall@10", k=10, mean=0.90, values=[])]
        )
        results = check_thresholds(run, {"nonexistent@5": 0.80})

        assert len(results) == 1
        assert results[0].actual == 0.0
        assert results[0].passed is False

    def test_empty_thresholds(self):
        run = _make_run_result(
            metrics=[MetricValue(name="recall@10", k=10, mean=0.90, values=[])]
        )
        results = check_thresholds(run, {})

        assert results == []
