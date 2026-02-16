"""Tests for evaluate CLI command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from berryeval.cli._app import app
from berryeval.config.types import DatasetMetadata, DatasetRecord
from berryeval.dataset.writer import write_dataset
from berryeval.persistence.types import LatencyReport, MetricValue, RunResult

if TYPE_CHECKING:
    from pathlib import Path

    from berryeval.config.eval_config import EvalConfig

runner = CliRunner()


class MockAdapter:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def close(self) -> None:
        return


def _write_test_dataset(filepath: Path) -> None:
    metadata = DatasetMetadata(
        config_hash="testhash123",
        model="gpt-4",
        chunk_size=800,
        overlap=100,
        timestamp="2026-01-01T00:00:00+00:00",
        corpus_stats={"files": 2, "total_chunks": 10},
        version="0.1.0",
        prompt_template="test: {chunk}",
    )
    records = [
        DatasetRecord(
            query="What is topic A?",
            relevant_chunk_ids=["doc1", "doc2"],
            chunk_text="Topic A content",
        ),
        DatasetRecord(
            query="What is topic B?",
            relevant_chunk_ids=["doc3"],
            chunk_text="Topic B content",
        ),
    ]
    write_dataset(filepath, metadata, records)


def _write_test_config(filepath: Path, per_query: bool = False) -> None:
    config = {
        "retriever": {
            "type": "mock",
            "param1": "value1",
        },
        "evaluation": {
            "k_values": [2],
            "metrics": ["recall", "precision", "mrr"],
            "per_query": per_query,
        },
    }
    filepath.write_text(yaml.safe_dump(config), encoding="utf-8")


def _make_mock_result(dataset_path: Path) -> RunResult:
    return RunResult(
        run_id="test-uuid-1234",
        timestamp="2026-01-01T00:00:00+00:00",
        config={"retriever": {"type": "mock"}, "evaluation": {"k_values": [2]}},
        dataset_path=str(dataset_path),
        dataset_metadata={"model": "gpt-4", "version": "0.1.0"},
        num_queries=2,
        metrics=[
            MetricValue(name="recall@2", k=2, mean=0.75, values=[]),
            MetricValue(name="precision@2", k=2, mean=0.5, values=[]),
            MetricValue(name="mrr@2", k=2, mean=0.7, values=[]),
        ],
        latency=LatencyReport(
            p50=10.0,
            p95=20.0,
            p99=25.0,
            mean=12.5,
            min_ms=5.0,
            max_ms=30.0,
            count=2,
        ),
        query_breakdowns=[],
    )


class TestEvaluateCommand:
    def test_evaluate_basic(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "eval.yaml"
        output_dir = tmp_path / "results"
        _write_test_dataset(dataset_path)
        _write_test_config(config_path)

        with (
            patch("berryeval.cli.evaluate.get_adapter_class", return_value=MockAdapter),
            patch("berryeval.cli.evaluate.EvaluationRunner") as runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = _make_mock_result(dataset_path)
            runner_cls.return_value = mock_runner

            result = runner.invoke(
                app,
                [
                    "evaluate",
                    str(dataset_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_dir),
                ],
            )

        assert result.exit_code == 0
        assert "Evaluation Complete" in result.output
        assert "Metrics" in result.output

    def test_evaluate_json_mode(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "eval.yaml"
        _write_test_dataset(dataset_path)
        _write_test_config(config_path)

        with (
            patch("berryeval.cli.evaluate.get_adapter_class", return_value=MockAdapter),
            patch("berryeval.cli.evaluate.EvaluationRunner") as runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = _make_mock_result(dataset_path)
            runner_cls.return_value = mock_runner

            result = runner.invoke(
                app,
                [
                    "--json",
                    "evaluate",
                    str(dataset_path),
                    "--config",
                    str(config_path),
                ],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["run_id"] == "test-uuid-1234"
        assert "metrics" in payload
        assert "latency" in payload

    def test_evaluate_per_query_flag(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "eval.yaml"
        _write_test_dataset(dataset_path)
        _write_test_config(config_path, per_query=False)

        with (
            patch("berryeval.cli.evaluate.get_adapter_class", return_value=MockAdapter),
            patch("berryeval.cli.evaluate.EvaluationRunner") as runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = _make_mock_result(dataset_path)
            runner_cls.return_value = mock_runner

            result = runner.invoke(
                app,
                [
                    "evaluate",
                    str(dataset_path),
                    "--config",
                    str(config_path),
                    "--per-query",
                ],
            )

            config_used: EvalConfig = runner_cls.call_args.args[1]

        assert result.exit_code == 0
        assert config_used.evaluation.per_query is True

    def test_evaluate_k_override(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "eval.yaml"
        _write_test_dataset(dataset_path)
        _write_test_config(config_path)

        with (
            patch("berryeval.cli.evaluate.get_adapter_class", return_value=MockAdapter),
            patch("berryeval.cli.evaluate.EvaluationRunner") as runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = _make_mock_result(dataset_path)
            runner_cls.return_value = mock_runner

            result = runner.invoke(
                app,
                [
                    "evaluate",
                    str(dataset_path),
                    "--config",
                    str(config_path),
                    "--k",
                    "3,7",
                ],
            )

            config_used: EvalConfig = runner_cls.call_args.args[1]

        assert result.exit_code == 0
        assert config_used.evaluation.k_values == [3, 7]

    def test_evaluate_missing_dataset(self, tmp_path: Path):
        config_path = tmp_path / "eval.yaml"
        _write_test_config(config_path)

        result = runner.invoke(
            app,
            [
                "evaluate",
                str(tmp_path / "missing.jsonl"),
                "--config",
                str(config_path),
            ],
        )

        assert result.exit_code != 0

    def test_evaluate_missing_config(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        _write_test_dataset(dataset_path)

        result = runner.invoke(
            app,
            [
                "evaluate",
                str(dataset_path),
                "--config",
                str(tmp_path / "missing.yaml"),
            ],
        )

        assert result.exit_code != 0

    def test_evaluate_config_validation_error(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "bad.yaml"
        _write_test_dataset(dataset_path)
        config_path.write_text("retriever: {}\n", encoding="utf-8")

        result = runner.invoke(
            app,
            ["evaluate", str(dataset_path), "--config", str(config_path)],
        )

        assert result.exit_code != 0
        assert "Config error" in result.output

    def test_evaluate_saves_results(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "eval.yaml"
        output_dir = tmp_path / "results"
        _write_test_dataset(dataset_path)
        _write_test_config(config_path)

        with (
            patch("berryeval.cli.evaluate.get_adapter_class", return_value=MockAdapter),
            patch("berryeval.cli.evaluate.EvaluationRunner") as runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = _make_mock_result(dataset_path)
            runner_cls.return_value = mock_runner

            result = runner.invoke(
                app,
                [
                    "evaluate",
                    str(dataset_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_dir),
                ],
            )

        assert result.exit_code == 0
        files = list(output_dir.glob("berryeval_run_*.json"))
        assert len(files) == 1

    def test_evaluate_custom_output_dir(self, tmp_path: Path):
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "eval.yaml"
        output_dir = tmp_path / "custom_results"
        _write_test_dataset(dataset_path)
        _write_test_config(config_path)

        with (
            patch("berryeval.cli.evaluate.get_adapter_class", return_value=MockAdapter),
            patch("berryeval.cli.evaluate.EvaluationRunner") as runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = _make_mock_result(dataset_path)
            runner_cls.return_value = mock_runner

            result = runner.invoke(
                app,
                [
                    "evaluate",
                    str(dataset_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_dir),
                ],
            )

        assert result.exit_code == 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("berryeval_run_*.json"))) == 1


class TestEvaluateThresholds:
    """Tests for the --fail-below threshold feature on evaluate."""

    def _invoke_with_fail_below(
        self,
        tmp_path: Path,
        fail_below: str,
        *,
        json_mode: bool = False,
    ):
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "eval.yaml"
        _write_test_dataset(dataset_path)
        _write_test_config(config_path)

        with (
            patch("berryeval.cli.evaluate.get_adapter_class", return_value=MockAdapter),
            patch("berryeval.cli.evaluate.EvaluationRunner") as runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = _make_mock_result(dataset_path)
            runner_cls.return_value = mock_runner

            args: list[str] = []
            if json_mode:
                args.append("--json")
            args.extend(
                [
                    "evaluate",
                    str(dataset_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(tmp_path / "results"),
                    "--fail-below",
                    fail_below,
                ]
            )

            return runner.invoke(app, args)

    def test_evaluate_fail_below_pass(self, tmp_path: Path) -> None:
        result = self._invoke_with_fail_below(tmp_path, "recall@2=0.60")
        assert result.exit_code == 0

    def test_evaluate_fail_below_fail(self, tmp_path: Path) -> None:
        result = self._invoke_with_fail_below(tmp_path, "recall@2=0.90")
        assert result.exit_code == 1

    def test_evaluate_fail_below_json_output(self, tmp_path: Path) -> None:
        result = self._invoke_with_fail_below(tmp_path, "recall@2=0.90", json_mode=True)
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert "threshold_results" in payload
        assert payload["thresholds_passed"] is False

    def test_evaluate_fail_below_pass_json(self, tmp_path: Path) -> None:
        result = self._invoke_with_fail_below(tmp_path, "recall@2=0.60", json_mode=True)
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["thresholds_passed"] is True

    def test_evaluate_fail_below_invalid_format(self, tmp_path: Path) -> None:
        result = self._invoke_with_fail_below(tmp_path, "bad")
        assert result.exit_code != 0

    def test_evaluate_no_fail_below(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        config_path = tmp_path / "eval.yaml"
        _write_test_dataset(dataset_path)
        _write_test_config(config_path)

        with (
            patch("berryeval.cli.evaluate.get_adapter_class", return_value=MockAdapter),
            patch("berryeval.cli.evaluate.EvaluationRunner") as runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = _make_mock_result(dataset_path)
            runner_cls.return_value = mock_runner

            result = runner.invoke(
                app,
                [
                    "evaluate",
                    str(dataset_path),
                    "--config",
                    str(config_path),
                    "--output",
                    str(tmp_path / "results"),
                ],
            )

        assert result.exit_code == 0
