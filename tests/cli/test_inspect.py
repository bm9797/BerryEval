"""Tests for the inspect CLI command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from berryeval.cli._app import app
from berryeval.config.types import DatasetMetadata, DatasetRecord
from berryeval.dataset.writer import write_dataset

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _make_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        config_hash="abc123def456ghi789jkl012mno345pq",
        model="gpt-4",
        chunk_size=800,
        overlap=100,
        timestamp="2026-01-01T00:00:00Z",
        corpus_stats={"files": 5, "total_chunks": 42},
        version="0.1.0",
        prompt_template="Generate a query for: {chunk}",
    )


def _make_records(n: int) -> list[DatasetRecord]:
    return [
        DatasetRecord(
            query=f"What is topic {i}?",
            relevant_chunk_ids=[f"doc0000_chunk{i:04d}"],
            chunk_text=f"This is the text of chunk {i}.",
            metadata={"source": "test"},
        )
        for i in range(n)
    ]


def _write_test_dataset(filepath: Path, num_records: int = 3) -> None:
    """Write a valid test JSONL dataset."""
    metadata = _make_metadata()
    records = _make_records(num_records)
    write_dataset(filepath, metadata, records)


class TestInspectCommand:
    def test_inspect_valid_dataset(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        _write_test_dataset(filepath)

        result = runner.invoke(app, ["inspect", str(filepath)])
        assert result.exit_code == 0
        assert "gpt-4" in result.output
        assert "800" in result.output
        assert "100" in result.output
        assert "0.1.0" in result.output

    def test_inspect_json_mode(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        _write_test_dataset(filepath, num_records=5)

        result = runner.invoke(app, ["--json", "inspect", str(filepath)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["model"] == "gpt-4"
        assert data["chunk_size"] == 800
        assert data["overlap"] == 100
        assert data["version"] == "0.1.0"
        assert data["query_count"] == 5
        assert data["corpus_stats"]["files"] == 5
        assert data["config_hash"] == "abc123def456ghi789jkl012mno345pq"

    def test_inspect_nonexistent_file(self, tmp_path: Path):
        filepath = tmp_path / "nonexistent.jsonl"

        result = runner.invoke(app, ["inspect", str(filepath)])
        assert result.exit_code == 1

    def test_inspect_invalid_file(self, tmp_path: Path):
        filepath = tmp_path / "bad.jsonl"
        filepath.write_text("this is not valid jsonl\n", encoding="utf-8")

        result = runner.invoke(app, ["inspect", str(filepath)])
        assert result.exit_code == 1

    def test_inspect_empty_dataset(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        _write_test_dataset(filepath, num_records=0)

        result = runner.invoke(app, ["--json", "inspect", str(filepath)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["query_count"] == 0

    def test_inspect_query_count(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        _write_test_dataset(filepath, num_records=10)

        result = runner.invoke(app, ["--json", "inspect", str(filepath)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["query_count"] == 10
