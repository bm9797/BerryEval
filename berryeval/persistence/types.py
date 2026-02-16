"""Pydantic models for persisted evaluation results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LatencyReport(BaseModel):
    """Latency report in milliseconds."""

    p50: float
    p95: float
    p99: float
    mean: float
    min_ms: float
    max_ms: float
    count: int


class MetricValue(BaseModel):
    """Aggregate metric value with optional per-query values."""

    name: str
    k: int
    mean: float
    values: list[float] = Field(default_factory=list)


class QueryBreakdown(BaseModel):
    """Per-query retrieval and score details."""

    query: str
    relevant_ids: list[str]
    retrieved_ids: list[str]
    scores: dict[str, float]
    latency_ms: float


class RunResult(BaseModel):
    """Complete result payload for one evaluation run."""

    run_id: str
    timestamp: str
    config: dict[str, object]
    dataset_path: str
    dataset_metadata: dict[str, object]
    num_queries: int
    metrics: list[MetricValue]
    latency: LatencyReport
    query_breakdowns: list[QueryBreakdown] = Field(default_factory=list)
