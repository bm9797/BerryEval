"""Dataset reader for JSONL files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from berryeval.config.types import DatasetMetadata, DatasetRecord

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


def read_dataset_metadata(filepath: Path) -> DatasetMetadata:
    """Read the metadata header from a dataset JSONL file.

    Raises:
        ValueError: If the first line is not a metadata record.
    """
    with open(filepath, encoding="utf-8") as f:
        first_line = f.readline()

    obj = json.loads(first_line)
    if obj.get("_type") != "metadata":
        msg = "first line of dataset is not a metadata record"
        raise ValueError(msg)

    return DatasetMetadata(
        config_hash=obj["config_hash"],
        model=obj["model"],
        chunk_size=obj["chunk_size"],
        overlap=obj["overlap"],
        timestamp=obj["timestamp"],
        corpus_stats=obj["corpus_stats"],
        version=obj["version"],
        prompt_template=obj["prompt_template"],
    )


def read_dataset_records(
    filepath: Path,
) -> Generator[DatasetRecord, None, None]:
    """Yield dataset records from a JSONL file, skipping the metadata line."""
    with open(filepath, encoding="utf-8") as f:
        next(f)  # skip metadata line
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield DatasetRecord(
                query=obj["query"],
                relevant_chunk_ids=obj["relevant_chunk_ids"],
                chunk_text=obj["chunk_text"],
                metadata=obj.get("metadata", {}),
            )


def count_records(filepath: Path) -> int:
    """Count the number of records in a dataset file (excludes metadata line)."""
    with open(filepath, encoding="utf-8") as f:
        # Subtract 1 for the metadata line
        return sum(1 for _ in f) - 1
