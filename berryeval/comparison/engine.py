"""Comparison engine for evaluating two RunResult objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from berryeval.comparison.types import ComparisonResult, MetricDelta, QueryRegression

if TYPE_CHECKING:
    from berryeval.persistence.types import RunResult

UNCHANGED_TOLERANCE = 1e-6


def _compute_direction(absolute_delta: float) -> str:
    if abs(absolute_delta) < UNCHANGED_TOLERANCE:
        return "unchanged"
    if absolute_delta > 0:
        return "improved"
    return "regressed"


def _compute_relative_delta_pct(baseline: float, candidate: float) -> float:
    if abs(baseline) > 1e-12:
        return ((candidate - baseline) / baseline) * 100.0
    if abs(candidate) < 1e-12:
        return 0.0
    return float("inf")


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    *,
    per_query: bool = False,
) -> ComparisonResult:
    """Compare two evaluation runs and produce a structured delta report."""
    warnings: list[str] = []

    if baseline.dataset_path != candidate.dataset_path:
        warnings.append(
            f"Dataset path mismatch: '{baseline.dataset_path}' vs '{candidate.dataset_path}'"
        )

    if baseline.num_queries != candidate.num_queries:
        warnings.append(
            f"Query count mismatch: {baseline.num_queries} vs {candidate.num_queries}"
        )

    if baseline.config != candidate.config:
        warnings.append("Configuration mismatch between baseline and candidate")

    baseline_metrics = {mv.name: mv.mean for mv in baseline.metrics}
    candidate_metrics = {mv.name: mv.mean for mv in candidate.metrics}

    shared_metrics = sorted(set(baseline_metrics) & set(candidate_metrics))

    metric_deltas: list[MetricDelta] = []
    for name in shared_metrics:
        b_val = baseline_metrics[name]
        c_val = candidate_metrics[name]
        abs_delta = c_val - b_val
        metric_deltas.append(
            MetricDelta(
                metric_name=name,
                baseline_value=b_val,
                candidate_value=c_val,
                absolute_delta=abs_delta,
                relative_delta_pct=_compute_relative_delta_pct(b_val, c_val),
                direction=_compute_direction(abs_delta),
            )
        )

    query_regressions: list[QueryRegression] = []
    if per_query and baseline.query_breakdowns and candidate.query_breakdowns:
        baseline_scores: dict[str, dict[str, float]] = {
            qb.query: qb.scores for qb in baseline.query_breakdowns
        }
        candidate_scores: dict[str, dict[str, float]] = {
            qb.query: qb.scores for qb in candidate.query_breakdowns
        }

        shared_queries = set(baseline_scores) & set(candidate_scores)
        for query in shared_queries:
            b_scores = baseline_scores[query]
            c_scores = candidate_scores[query]
            shared_score_names = set(b_scores) & set(c_scores)
            for metric_name in shared_score_names:
                b_score = b_scores[metric_name]
                c_score = c_scores[metric_name]
                delta = c_score - b_score
                if delta < 0:
                    query_regressions.append(
                        QueryRegression(
                            query=query,
                            metric_name=metric_name,
                            baseline_score=b_score,
                            candidate_score=c_score,
                            delta=delta,
                        )
                    )

        query_regressions.sort(key=lambda r: r.delta)

    return ComparisonResult(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_num_queries=baseline.num_queries,
        candidate_num_queries=candidate.num_queries,
        metric_deltas=metric_deltas,
        query_regressions=query_regressions,
        warnings=warnings,
    )
