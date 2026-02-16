"""Latency tracking utilities for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LatencyStats:
    """Aggregate latency statistics in seconds."""

    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    count: int


class LatencyTracker:
    """Collect and summarize per-query latencies."""

    def __init__(self) -> None:
        self._latencies: list[float] = []

    def record(self, duration: float) -> None:
        self._latencies.append(duration)

    def get_latencies(self) -> list[float]:
        return list(self._latencies)

    def compute_stats(self) -> LatencyStats:
        if not self._latencies:
            msg = "No latencies recorded"
            raise ValueError(msg)

        arr = np.array(self._latencies, dtype=np.float64)
        return LatencyStats(
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            mean=float(np.mean(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            count=len(self._latencies),
        )
