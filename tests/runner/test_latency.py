"""Tests for latency tracking."""

from __future__ import annotations

import pytest

from berryeval.runner.latency import LatencyStats, LatencyTracker


class TestLatencyTracker:
    def test_record_single(self):
        tracker = LatencyTracker()
        tracker.record(0.123)
        assert tracker.get_latencies() == [0.123]

    def test_record_multiple(self):
        tracker = LatencyTracker()
        tracker.record(0.1)
        tracker.record(0.2)
        tracker.record(0.3)
        assert tracker.get_latencies() == [0.1, 0.2, 0.3]

    def test_compute_stats(self):
        tracker = LatencyTracker()
        samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for value in samples:
            tracker.record(value)

        stats = tracker.compute_stats()
        assert stats.p50 == pytest.approx(0.55)
        assert stats.p95 == pytest.approx(0.955)
        assert stats.p99 == pytest.approx(0.991)
        assert stats.mean == pytest.approx(0.55)
        assert stats.min == pytest.approx(0.1)
        assert stats.max == pytest.approx(1.0)
        assert stats.count == 10

    def test_empty_tracker_raises(self):
        tracker = LatencyTracker()
        with pytest.raises(ValueError, match="No latencies"):
            tracker.compute_stats()

    def test_get_latencies_returns_copy(self):
        tracker = LatencyTracker()
        tracker.record(0.2)
        latencies = tracker.get_latencies()
        latencies.append(0.5)
        assert tracker.get_latencies() == [0.2]


class TestLatencyStats:
    def test_fields_accessible(self):
        stats = LatencyStats(
            p50=0.1,
            p95=0.2,
            p99=0.3,
            mean=0.15,
            min=0.05,
            max=0.4,
            count=5,
        )
        assert stats.p50 == 0.1
        assert stats.count == 5
