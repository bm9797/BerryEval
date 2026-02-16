"""Evaluation YAML configuration models and loader."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")
_ALLOWED_METRICS = {"recall", "precision", "mrr", "ndcg", "hit_rate"}


class RetrieverConfig(BaseModel):
    """Retriever adapter config section."""

    model_config = ConfigDict(extra="allow")

    type: str


class EvaluationSettings(BaseModel):
    """Evaluation settings section."""

    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    metrics: list[str] = Field(
        default_factory=lambda: ["recall", "precision", "mrr", "ndcg", "hit_rate"]
    )
    per_query: bool = False

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, value: list[int]) -> list[int]:
        if any(k < 1 for k in value):
            msg = "All k_values must be >= 1"
            raise ValueError(msg)
        return value

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        invalid = [metric for metric in value if metric not in _ALLOWED_METRICS]
        if invalid:
            allowed = ", ".join(sorted(_ALLOWED_METRICS))
            msg = (
                "Invalid metric name(s): "
                f"{', '.join(invalid)}. Allowed metrics: {allowed}"
            )
            raise ValueError(msg)
        return value


class EvalConfig(BaseModel):
    """Top-level evaluation config."""

    retriever: RetrieverConfig
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} placeholders in a string value."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            msg = (
                f"Environment variable '{var_name}' is not set "
                "(referenced in config)"
            )
            raise ValueError(msg)
        return env_value

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _substitute_env_vars_recursive(obj: Any) -> Any:
    """Apply environment variable substitution across nested config structures."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {key: _substitute_env_vars_recursive(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env_vars_recursive(item) for item in obj]
    return obj


def load_eval_config(filepath: Path) -> EvalConfig:
    """Load and validate an evaluation config YAML file."""
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    with open(filepath, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    substituted = _substitute_env_vars_recursive(raw_data)
    return EvalConfig.model_validate(substituted)
