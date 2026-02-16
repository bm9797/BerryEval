"""Dataset writer for JSONL output."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from berryeval.config.types import DatasetMetadata, DatasetRecord


def write_dataset(
    filepath: Path,
    metadata: DatasetMetadata,
    records: list[DatasetRecord],
) -> None:
    """Write a dataset to a JSONL file.

    The first line contains the metadata object, followed by one line
    per record. All lines use compact, deterministic JSON encoding.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        meta_obj: dict[str, object] = {"_type": "metadata", **metadata.to_dict()}
        f.write(json.dumps(meta_obj, separators=(",", ":"), sort_keys=True))
        f.write("\n")
        for record in records:
            rec_obj: dict[str, object] = {"_type": "record", **record.to_dict()}
            f.write(json.dumps(rec_obj, separators=(",", ":"), sort_keys=True))
            f.write("\n")
