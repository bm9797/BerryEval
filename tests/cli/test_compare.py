"""Tests for compare CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from berryeval.cli._app import app
from berryeval.persistence.types import (
    LatencyReport,
    MetricValue,
    QueryBreakdown,
    RunResult,
)

runner = CliRunner()

_DEFAULT_LATENCY = LatencyReport(
    p50=10.0, p95=20.0, p99=25.0, mean=12.5, min_ms=5.0, max_ms=30.0, count=2
)


def _make_run_result(
    run_id: str = "baseline-1234",
    dataset_path: str = "/data/test.jsonl",
    num_queries: int = 2,
    metrics: list[MetricValue] | None = None,
    query_breakdowns: list[QueryBreakdown] | None = None,
    config: dict[str, Any] | None = None,
) -> RunResult:
    if metrics is None:
        metrics = [MetricValue(name="recall@10", k=10, mean=0.80, values=[])]
    return RunResult(
        run_id=run_id,
        timestamp="2026-01-01T00:00:00+00:00",
        config=config or {"retriever": {"type": "mock"}},
        dataset_path=dataset_path,
        dataset_metadata={"model": "gpt-4"},
        num_queries=num_queries,
        metrics=metrics,
        latency=_DEFAULT_LATENCY,
        query_breakdowns=query_breakdowns or [],
    )


def _load_side_effect(baseline: RunResult, candidate: RunResult):
    """Return a side_effect function that maps path to result."""

    def _loader(filepath: Path) -> RunResult:
        if "baseline" in str(filepath):
            return baseline
        return candidate

    return _loader


class TestCompareCommand:
    def test_compare_basic(self) -> None:
        baseline = _make_run_result(
            run_id="baseline-1234",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.70, values=[])],
        )
        candidate = _make_run_result(
            run_id="candidate-5678",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.85, values=[])],
        )

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app, ["compare", "/tmp/baseline.json", "/tmp/candidate.json"]
            )

        assert result.exit_code == 0
        assert "IMPROVED" in result.output

    def test_compare_regression_detected(self) -> None:
        baseline = _make_run_result(
            run_id="baseline-1234",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.90, values=[])],
        )
        candidate = _make_run_result(
            run_id="candidate-5678",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.70, values=[])],
        )

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app, ["compare", "/tmp/baseline.json", "/tmp/candidate.json"]
            )

        assert result.exit_code == 0
        assert "REGRESSED" in result.output

    def test_compare_json_mode(self) -> None:
        baseline = _make_run_result(
            run_id="baseline-1234",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.70, values=[])],
        )
        candidate = _make_run_result(
            run_id="candidate-5678",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.85, values=[])],
        )

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app,
                ["--json", "compare", "/tmp/baseline.json", "/tmp/candidate.json"],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "metric_deltas" in payload

    def test_compare_fail_below_pass(self) -> None:
        baseline = _make_run_result(run_id="baseline-1234")
        candidate = _make_run_result(
            run_id="candidate-5678",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.85, values=[])],
        )

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app,
                [
                    "compare",
                    "/tmp/baseline.json",
                    "/tmp/candidate.json",
                    "--fail-below",
                    "recall@10=0.80",
                ],
            )

        assert result.exit_code == 0

    def test_compare_fail_below_fail(self) -> None:
        baseline = _make_run_result(run_id="baseline-1234")
        candidate = _make_run_result(
            run_id="candidate-5678",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.70, values=[])],
        )

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app,
                [
                    "compare",
                    "/tmp/baseline.json",
                    "/tmp/candidate.json",
                    "--fail-below",
                    "recall@10=0.80",
                ],
            )

        assert result.exit_code == 1

    def test_compare_fail_below_json_mode(self) -> None:
        baseline = _make_run_result(run_id="baseline-1234")
        candidate = _make_run_result(
            run_id="candidate-5678",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.70, values=[])],
        )

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app,
                [
                    "--json",
                    "compare",
                    "/tmp/baseline.json",
                    "/tmp/candidate.json",
                    "--fail-below",
                    "recall@10=0.80",
                ],
            )

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["thresholds_passed"] is False

    def test_compare_per_query(self) -> None:
        baseline = _make_run_result(
            run_id="baseline-1234",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.80, values=[])],
            query_breakdowns=[
                QueryBreakdown(
                    query="What is A?",
                    relevant_ids=["d1"],
                    retrieved_ids=["d1", "d2"],
                    scores={"recall@10": 1.0},
                    latency_ms=10.0,
                ),
            ],
        )
        candidate = _make_run_result(
            run_id="candidate-5678",
            metrics=[MetricValue(name="recall@10", k=10, mean=0.60, values=[])],
            query_breakdowns=[
                QueryBreakdown(
                    query="What is A?",
                    relevant_ids=["d1"],
                    retrieved_ids=["d2"],
                    scores={"recall@10": 0.0},
                    latency_ms=12.0,
                ),
            ],
        )

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app,
                [
                    "compare",
                    "/tmp/baseline.json",
                    "/tmp/candidate.json",
                    "--per-query",
                ],
            )

        assert result.exit_code == 0
        assert "Regression" in result.output or "What is A?" in result.output

    def test_compare_invalid_threshold_format(self) -> None:
        baseline = _make_run_result(run_id="baseline-1234")
        candidate = _make_run_result(run_id="candidate-5678")

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app,
                [
                    "compare",
                    "/tmp/baseline.json",
                    "/tmp/candidate.json",
                    "--fail-below",
                    "bad",
                ],
            )

        assert result.exit_code != 0
        assert "Invalid threshold" in result.output

    def test_compare_missing_baseline_file(self) -> None:
        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=FileNotFoundError("/tmp/baseline.json"),
        ):
            result = runner.invoke(
                app, ["compare", "/tmp/baseline.json", "/tmp/candidate.json"]
            )

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_compare_warnings_displayed(self) -> None:
        baseline = _make_run_result(
            run_id="baseline-1234",
            dataset_path="/data/test_a.jsonl",
        )
        candidate = _make_run_result(
            run_id="candidate-5678",
            dataset_path="/data/test_b.jsonl",
        )

        with patch(
            "berryeval.cli.compare.load_result",
            side_effect=_load_side_effect(baseline, candidate),
        ):
            result = runner.invoke(
                app, ["compare", "/tmp/baseline.json", "/tmp/candidate.json"]
            )

        assert result.exit_code == 0
        assert "Warning" in result.output
