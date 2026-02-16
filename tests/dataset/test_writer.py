"""Tests for berryeval.dataset.writer and berryeval.dataset.reader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from berryeval.config.types import DatasetMetadata, DatasetRecord
from berryeval.dataset.reader import (
    count_records,
    read_dataset_metadata,
    read_dataset_records,
)
from berryeval.dataset.writer import write_dataset

if TYPE_CHECKING:
    from pathlib import Path


def _make_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        config_hash="abc123",
        model="gpt-4",
        chunk_size=800,
        overlap=100,
        timestamp="2026-01-01T00:00:00Z",
        corpus_stats={"total_files": 5, "total_chunks": 42},
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


class TestWriteAndRead:
    def test_write_and_read_metadata(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        meta = _make_metadata()
        write_dataset(filepath, meta, [])
        loaded = read_dataset_metadata(filepath)
        assert loaded.config_hash == meta.config_hash
        assert loaded.model == meta.model
        assert loaded.chunk_size == meta.chunk_size
        assert loaded.overlap == meta.overlap
        assert loaded.timestamp == meta.timestamp
        assert loaded.corpus_stats == meta.corpus_stats
        assert loaded.version == meta.version
        assert loaded.prompt_template == meta.prompt_template

    def test_write_and_read_records(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        records = _make_records(5)
        write_dataset(filepath, _make_metadata(), records)
        loaded = list(read_dataset_records(filepath))
        assert len(loaded) == 5
        for orig, read in zip(records, loaded, strict=True):
            assert read.query == orig.query
            assert read.relevant_chunk_ids == orig.relevant_chunk_ids
            assert read.chunk_text == orig.chunk_text
            assert read.metadata == orig.metadata

    def test_jsonl_format(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        write_dataset(filepath, _make_metadata(), _make_records(3))
        lines = filepath.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 4  # 1 metadata + 3 records
        for line in lines:
            obj = json.loads(line)
            assert "_type" in obj
        assert json.loads(lines[0])["_type"] == "metadata"
        for line in lines[1:]:
            assert json.loads(line)["_type"] == "record"

    def test_record_count(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        write_dataset(filepath, _make_metadata(), _make_records(7))
        assert count_records(filepath) == 7

    def test_empty_dataset(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        write_dataset(filepath, _make_metadata(), [])
        assert count_records(filepath) == 0
        loaded = list(read_dataset_records(filepath))
        assert loaded == []

    def test_portable_encoding(self, tmp_path: Path):
        filepath = tmp_path / "dataset.jsonl"
        record = DatasetRecord(
            query="Was ist das Thema?",
            relevant_chunk_ids=["doc0000_chunk0000"],
            chunk_text="Ubersicht uber die Architektur.",
            metadata={"lang": "de"},
        )
        write_dataset(filepath, _make_metadata(), [record])
        loaded = list(read_dataset_records(filepath))
        assert loaded[0].query == record.query
        assert loaded[0].chunk_text == record.chunk_text

    def test_round_trip(self, tmp_path: Path):
        filepath1 = tmp_path / "dataset1.jsonl"
        filepath2 = tmp_path / "dataset2.jsonl"
        meta = _make_metadata()
        records = _make_records(5)
        write_dataset(filepath1, meta, records)

        # Read back and write again
        loaded_meta = read_dataset_metadata(filepath1)
        loaded_records = list(read_dataset_records(filepath1))
        write_dataset(filepath2, loaded_meta, loaded_records)

        assert filepath1.read_text(encoding="utf-8") == filepath2.read_text(
            encoding="utf-8"
        )
