"""Pydantic models for run comparison results."""

from __future__ import annotations

from pydantic import BaseModel


class MetricDelta(BaseModel):
    """Change in a single metric between baseline and candidate runs."""

    metric_name: str
    baseline_value: float
    candidate_value: float
    absolute_delta: float
    relative_delta_pct: float
    direction: str  # "improved", "regressed", "unchanged"


class QueryRegression(BaseModel):
    """A single query where the candidate scored worse than the baseline."""

    query: str
    metric_name: str
    baseline_score: float
    candidate_score: float
    delta: float  # candidate - baseline, negative = regression


class ComparisonResult(BaseModel):
    """Full comparison between a baseline and candidate run."""

    baseline_run_id: str
    candidate_run_id: str
    baseline_num_queries: int
    candidate_num_queries: int
    metric_deltas: list[MetricDelta]
    query_regressions: list[QueryRegression]
    warnings: list[str]


class ThresholdResult(BaseModel):
    """Whether a single metric meets its threshold."""

    metric_name: str
    threshold: float
    actual: float
    passed: bool
