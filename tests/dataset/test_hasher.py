"""Tests for berryeval.dataset.hasher."""

from __future__ import annotations

from typing import TYPE_CHECKING

from berryeval.dataset.hasher import compute_config_hash, compute_file_hash

if TYPE_CHECKING:
    from pathlib import Path


class TestComputeConfigHash:
    def test_deterministic(self):
        config = {"model": "gpt-4", "chunk_size": 800}
        h1 = compute_config_hash(config)
        h2 = compute_config_hash(config)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_key_order_irrelevant(self):
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_values(self):
        h1 = compute_config_hash({"model": "gpt-4"})
        h2 = compute_config_hash({"model": "gpt-3.5"})
        assert h1 != h2


class TestComputeFileHash:
    def test_file_hash(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        h = compute_file_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_file_hash_deterministic(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("deterministic content", encoding="utf-8")
        h1 = compute_file_hash(f)
        h2 = compute_file_hash(f)
        assert h1 == h2

    def test_file_hash_different_content(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A", encoding="utf-8")
        f2.write_text("content B", encoding="utf-8")
        assert compute_file_hash(f1) != compute_file_hash(f2)
