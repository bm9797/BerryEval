"""Tests for evaluation config models and loader."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml
from pydantic import ValidationError

from berryeval.config.eval_config import (
    EvalConfig,
    EvaluationSettings,
    RetrieverConfig,
    _substitute_env_vars,
    _substitute_env_vars_recursive,
    load_eval_config,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestRetrieverConfig:
    def test_valid_config(self):
        cfg = RetrieverConfig(type="pinecone")
        assert cfg.type == "pinecone"

    def test_extra_fields_allowed(self):
        cfg = RetrieverConfig(type="pinecone", api_key="x", index_name="idx")
        dumped = cfg.model_dump()
        assert dumped["api_key"] == "x"
        assert dumped["index_name"] == "idx"

    def test_missing_type_fails(self):
        with pytest.raises(ValidationError):
            RetrieverConfig()  # type: ignore[call-arg]


class TestEvaluationSettings:
    def test_defaults(self):
        settings = EvaluationSettings()
        assert settings.k_values == [5, 10, 20]
        assert settings.metrics == ["recall", "precision", "mrr", "ndcg", "hit_rate"]
        assert settings.per_query is False

    def test_custom_k_values(self):
        settings = EvaluationSettings(k_values=[1, 3, 5])
        assert settings.k_values == [1, 3, 5]

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValidationError, match="Invalid metric"):
            EvaluationSettings(metrics=["recall", "not_a_metric"])

    def test_invalid_k_rejected(self):
        with pytest.raises(ValidationError, match=">= 1"):
            EvaluationSettings(k_values=[0, 5])


class TestEvalConfig:
    def test_full_config(self):
        cfg = EvalConfig(
            retriever={"type": "pinecone", "api_key": "x", "index_name": "test"},
            evaluation={"k_values": [2], "metrics": ["recall"], "per_query": True},
        )
        assert cfg.retriever.type == "pinecone"
        assert cfg.evaluation.k_values == [2]
        assert cfg.evaluation.metrics == ["recall"]
        assert cfg.evaluation.per_query is True

    def test_default_evaluation_settings(self):
        cfg = EvalConfig(retriever={"type": "pinecone"})
        assert cfg.evaluation.k_values == [5, 10, 20]


class TestEnvVarSubstitution:
    def test_single_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_API_KEY", "abc123")
        assert _substitute_env_vars("${TEST_API_KEY}") == "abc123"

    def test_multiple_vars_in_single_string(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HOST", "example.com")
        monkeypatch.setenv("PORT", "443")
        value = "https://${HOST}:${PORT}/index"
        assert _substitute_env_vars(value) == "https://example.com:443/index"

    def test_nested_dict_substitution(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("API_KEY", "secret")
        data: dict[str, Any] = {
            "retriever": {"type": "pinecone", "api_key": "${API_KEY}"},
            "evaluation": {"k_values": [5]},
        }
        substituted = _substitute_env_vars_recursive(data)
        assert substituted["retriever"]["api_key"] == "secret"

    def test_missing_var_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ValueError, match="MISSING_VAR"):
            _substitute_env_vars("${MISSING_VAR}")

    def test_non_string_values_pass_through(self):
        data: dict[str, Any] = {"a": 1, "b": True, "c": [1, 2], "d": {"x": None}}
        substituted = _substitute_env_vars_recursive(data)
        assert substituted == data


class TestLoadEvalConfig:
    def test_load_valid_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TEST_API_KEY", "key-123")
        filepath = tmp_path / "eval.yaml"
        filepath.write_text(
            "\n".join(
                [
                    "retriever:",
                    "  type: pinecone",
                    "  api_key: ${TEST_API_KEY}",
                    "  index_name: my-index",
                    "evaluation:",
                    "  k_values: [5, 10]",
                    "  metrics: [recall, precision]",
                    "  per_query: true",
                ]
            ),
            encoding="utf-8",
        )

        cfg = load_eval_config(filepath)
        assert cfg.retriever.type == "pinecone"
        assert cfg.retriever.model_dump()["api_key"] == "key-123"
        assert cfg.evaluation.k_values == [5, 10]
        assert cfg.evaluation.metrics == ["recall", "precision"]
        assert cfg.evaluation.per_query is True

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_eval_config(tmp_path / "missing.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path):
        filepath = tmp_path / "bad.yaml"
        filepath.write_text("retriever: [", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_eval_config(filepath)

    def test_invalid_structure_raises_validation_error(self, tmp_path: Path):
        filepath = tmp_path / "invalid.yaml"
        filepath.write_text(
            "\n".join(
                [
                    "retriever:",
                    "  api_key: x",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            load_eval_config(filepath)
