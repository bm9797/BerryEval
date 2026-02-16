"""Tests for the comparison engine."""

from __future__ import annotations

import math

from berryeval.comparison.engine import compare_runs
from berryeval.persistence.types import (
    LatencyReport,
    MetricValue,
    QueryBreakdown,
    RunResult,
)


def _make_latency() -> LatencyReport:
    return LatencyReport(
        p50=10.0, p95=20.0, p99=30.0, mean=12.0, min_ms=5.0, max_ms=50.0, count=10
    )


def _make_run_result(
    *,
    run_id: str = "run-1",
    dataset_path: str = "data.jsonl",
    num_queries: int = 10,
    config: dict[str, object] | None = None,
    metrics: list[MetricValue] | None = None,
    query_breakdowns: list[QueryBreakdown] | None = None,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        timestamp="2026-01-01T00:00:00Z",
        config=config or {"model": "test"},
        dataset_path=dataset_path,
        dataset_metadata={},
        num_queries=num_queries,
        metrics=metrics or [],
        latency=_make_latency(),
        query_breakdowns=query_breakdowns or [],
    )


class TestCompareIdenticalRuns:
    def test_compare_identical_runs(self):
        metrics = [MetricValue(name="recall@10", k=10, mean=0.85, values=[])]
        baseline = _make_run_result(run_id="b", metrics=metrics)
        candidate = _make_run_result(run_id="c", metrics=metrics)

        result = compare_runs(baseline, candidate)

        assert len(result.metric_deltas) == 1
        delta = result.metric_deltas[0]
        assert delta.direction == "unchanged"
        assert delta.absolute_delta == 0.0
        assert delta.relative_delta_pct == 0.0
        assert result.warnings == []


class TestCompareImprovedMetrics:
    def test_compare_improved_metrics(self):
        baseline = _make_run_result(
            run_id="b",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.80, values=[])],
        )
        candidate = _make_run_result(
            run_id="c",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.90, values=[])],
        )

        result = compare_runs(baseline, candidate)

        delta = result.metric_deltas[0]
        assert delta.direction == "improved"
        assert delta.absolute_delta > 0
        assert delta.relative_delta_pct > 0


class TestCompareRegressedMetrics:
    def test_compare_regressed_metrics(self):
        baseline = _make_run_result(
            run_id="b",
            metrics=[MetricValue(name="precision@5", k=5, mean=0.90, values=[])],
        )
        candidate = _make_run_result(
            run_id="c",
            metrics=[MetricValue(name="precision@5", k=5, mean=0.70, values=[])],
        )

        result = compare_runs(baseline, candidate)

        delta = result.metric_deltas[0]
        assert delta.direction == "regressed"
        assert delta.absolute_delta < 0


class TestCompareMixedResults:
    def test_compare_mixed_results(self):
        baseline = _make_run_result(
            run_id="b",
            metrics=[
                MetricValue(name="recall@10", k=10, mean=0.80, values=[]),
                MetricValue(name="precision@5", k=5, mean=0.90, values=[]),
            ],
        )
        candidate = _make_run_result(
            run_id="c",
            metrics=[
                MetricValue(name="recall@10", k=10, mean=0.90, values=[]),
                MetricValue(name="precision@5", k=5, mean=0.70, values=[]),
            ],
        )

        result = compare_runs(baseline, candidate)

        directions = {d.metric_name: d.direction for d in result.metric_deltas}
        assert directions["recall@10"] == "improved"
        assert directions["precision@5"] == "regressed"


class TestCompareMissingMetrics:
    def test_compare_missing_metrics(self):
        baseline = _make_run_result(
            run_id="b",
            metrics=[
                MetricValue(name="recall@10", k=10, mean=0.80, values=[]),
                MetricValue(name="ndcg@10", k=10, mean=0.75, values=[]),
            ],
        )
        candidate = _make_run_result(
            run_id="c",
            metrics=[
                MetricValue(name="recall@10", k=10, mean=0.90, values=[]),
                MetricValue(name="mrr@10", k=10, mean=0.65, values=[]),
            ],
        )

        result = compare_runs(baseline, candidate)

        assert len(result.metric_deltas) == 1
        assert result.metric_deltas[0].metric_name == "recall@10"


class TestWarnings:
    def test_compare_different_datasets_warning(self):
        baseline = _make_run_result(run_id="b", dataset_path="a.jsonl")
        candidate = _make_run_result(run_id="c", dataset_path="b.jsonl")

        result = compare_runs(baseline, candidate)

        assert any("Dataset path mismatch" in w for w in result.warnings)

    def test_compare_different_configs_warning(self):
        baseline = _make_run_result(run_id="b", config={"model": "v1"})
        candidate = _make_run_result(run_id="c", config={"model": "v2"})

        result = compare_runs(baseline, candidate)

        assert any("Configuration mismatch" in w for w in result.warnings)

    def test_compare_different_query_counts_warning(self):
        baseline = _make_run_result(run_id="b", num_queries=10)
        candidate = _make_run_result(run_id="c", num_queries=20)

        result = compare_runs(baseline, candidate)

        assert any("Query count mismatch" in w for w in result.warnings)


class TestPerQueryRegressions:
    def test_compare_per_query_regressions(self):
        baseline = _make_run_result(
            run_id="b",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.80, values=[])],
            query_breakdowns=[
                QueryBreakdown(
                    query="q1",
                    relevant_ids=["a"],
                    retrieved_ids=["a"],
                    scores={"recall@10": 0.9},
                    latency_ms=10.0,
                ),
                QueryBreakdown(
                    query="q2",
                    relevant_ids=["b"],
                    retrieved_ids=["b"],
                    scores={"recall@10": 0.7},
                    latency_ms=10.0,
                ),
            ],
        )
        candidate = _make_run_result(
            run_id="c",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.75, values=[])],
            query_breakdowns=[
                QueryBreakdown(
                    query="q1",
                    relevant_ids=["a"],
                    retrieved_ids=["a"],
                    scores={"recall@10": 0.6},
                    latency_ms=10.0,
                ),
                QueryBreakdown(
                    query="q2",
                    relevant_ids=["b"],
                    retrieved_ids=["b"],
                    scores={"recall@10": 0.9},
                    latency_ms=10.0,
                ),
            ],
        )

        result = compare_runs(baseline, candidate, per_query=True)

        assert len(result.query_regressions) == 1
        reg = result.query_regressions[0]
        assert reg.query == "q1"
        assert reg.delta < 0

    def test_compare_per_query_no_breakdowns(self):
        baseline = _make_run_result(run_id="b", query_breakdowns=[])
        candidate = _make_run_result(run_id="c", query_breakdowns=[])

        result = compare_runs(baseline, candidate, per_query=True)

        assert result.query_regressions == []

    def test_compare_per_query_sorted_by_severity(self):
        baseline = _make_run_result(
            run_id="b",
            query_breakdowns=[
                QueryBreakdown(
                    query="q1",
                    relevant_ids=[],
                    retrieved_ids=[],
                    scores={"recall@10": 0.8},
                    latency_ms=10.0,
                ),
                QueryBreakdown(
                    query="q2",
                    relevant_ids=[],
                    retrieved_ids=[],
                    scores={"recall@10": 0.9},
                    latency_ms=10.0,
                ),
            ],
        )
        candidate = _make_run_result(
            run_id="c",
            query_breakdowns=[
                QueryBreakdown(
                    query="q1",
                    relevant_ids=[],
                    retrieved_ids=[],
                    scores={"recall@10": 0.7},
                    latency_ms=10.0,
                ),
                QueryBreakdown(
                    query="q2",
                    relevant_ids=[],
                    retrieved_ids=[],
                    scores={"recall@10": 0.5},
                    latency_ms=10.0,
                ),
            ],
        )

        result = compare_runs(baseline, candidate, per_query=True)

        assert len(result.query_regressions) == 2
        # Worst regression first (most negative delta)
        assert result.query_regressions[0].delta <= result.query_regressions[1].delta
        assert result.query_regressions[0].query == "q2"  # -0.4 < -0.1


class TestEdgeCases:
    def test_compare_zero_baseline(self):
        baseline = _make_run_result(
            run_id="b",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.0, values=[])],
        )
        candidate = _make_run_result(
            run_id="c",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.5, values=[])],
        )

        result = compare_runs(baseline, candidate)

        delta = result.metric_deltas[0]
        assert delta.relative_delta_pct == float("inf") or math.isinf(
            delta.relative_delta_pct
        )

    def test_compare_both_zero(self):
        baseline = _make_run_result(
            run_id="b",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.0, values=[])],
        )
        candidate = _make_run_result(
            run_id="c",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.0, values=[])],
        )

        result = compare_runs(baseline, candidate)

        delta = result.metric_deltas[0]
        assert delta.direction == "unchanged"
        assert delta.relative_delta_pct == 0.0
