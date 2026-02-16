"""JSON persistence for evaluation results."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from berryeval.persistence.types import RunResult

if TYPE_CHECKING:
    from pathlib import Path


def _timestamp_for_filename(timestamp: str) -> str:
    normalized = timestamp.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed.strftime("%Y%m%d_%H%M%S")


def save_result(result: RunResult, output_dir: Path) -> Path:
    """Save a run result as JSON in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = _timestamp_for_filename(result.timestamp)
    filename = f"berryeval_run_{stamp}_{result.run_id[:8]}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="python"), f, indent=2)

    return filepath


def load_result(filepath: Path) -> RunResult:
    """Load and validate a saved run result from JSON."""
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    return RunResult.model_validate(data)
