"""Hashing utilities for configuration and file integrity."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def compute_config_hash(config: dict[str, object]) -> str:
    """Compute a deterministic SHA-256 hash of a configuration dict.

    Keys are sorted and output is compact to ensure identical dicts
    always produce the same hash regardless of insertion order.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_hash(filepath: str | Path) -> str:
    """Compute the SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            block = f.read(8192)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
