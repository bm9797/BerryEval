"""Threshold parsing and checking for evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from berryeval.comparison.types import ThresholdResult

if TYPE_CHECKING:
    from berryeval.persistence.types import RunResult


def parse_thresholds(threshold_str: str) -> dict[str, float]:
    """Parse a threshold string like ``'recall@10=0.80,precision@5=0.60'``.

    Raises ``ValueError`` for invalid formats.
    """
    if not threshold_str or not threshold_str.strip():
        msg = "Threshold string must not be empty"
        raise ValueError(msg)

    result: dict[str, float] = {}
    for pair in threshold_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            msg = f"Invalid threshold pair (missing '='): '{pair}'"
            raise ValueError(msg)
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            msg = "Threshold metric name must not be empty"
            raise ValueError(msg)
        try:
            result[key] = float(value)
        except ValueError:
            msg = f"Non-numeric threshold value for '{key}': '{value}'"
            raise ValueError(msg) from None
    return result


def check_thresholds(
    result: RunResult,
    thresholds: dict[str, float],
) -> list[ThresholdResult]:
    """Check run metrics against thresholds.

    Unknown metrics get ``actual=0.0`` and ``passed=False``.
    """
    metric_lookup = {mv.name: mv.mean for mv in result.metrics}

    results: list[ThresholdResult] = []
    for metric_name, threshold in thresholds.items():
        actual = metric_lookup.get(metric_name, 0.0)
        passed = actual >= threshold
        results.append(
            ThresholdResult(
                metric_name=metric_name,
                threshold=threshold,
                actual=actual,
                passed=passed,
            )
        )
    return results
