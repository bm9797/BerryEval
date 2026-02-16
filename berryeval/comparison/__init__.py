"""Run comparison and threshold checking."""

from berryeval.comparison.engine import compare_runs
from berryeval.comparison.thresholds import check_thresholds, parse_thresholds
from berryeval.comparison.types import (
    ComparisonResult,
    MetricDelta,
    QueryRegression,
    ThresholdResult,
)

__all__ = [
    "ComparisonResult",
    "MetricDelta",
    "QueryRegression",
    "ThresholdResult",
    "check_thresholds",
    "compare_runs",
    "parse_thresholds",
]
