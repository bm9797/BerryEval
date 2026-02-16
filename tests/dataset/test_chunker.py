"""Tests for berryeval.dataset.chunker."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from berryeval.dataset.chunker import chunk_corpus, chunk_text


def _make_text(length: int) -> str:
    """Generate repeating words to fill the desired character count."""
    word = "hello "
    return (word * (length // len(word) + 1))[:length]


class TestChunkText:
    def test_basic_chunking(self):
        text = _make_text(2000)
        chunks = chunk_text(text, "test.txt", chunk_size=800, overlap=100)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 800
        # All text should be covered
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(text)

    def test_word_boundary_respect(self):
        text = _make_text(2000)
        chunks = chunk_text(text, "test.txt", chunk_size=800, overlap=100)
        for chunk in chunks[:-1]:
            # Non-final chunks should split at a word boundary: the character
            # at end_char in the original text should be a space.
            if chunk.end_char < len(text):
                assert text[chunk.end_char] == " "

    def test_overlap_produces_shared_text(self):
        text = _make_text(2000)
        chunks = chunk_text(text, "test.txt", chunk_size=800, overlap=100)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            curr_end = chunks[i].end_char
            next_start = chunks[i + 1].start_char
            # Next chunk starts before current one ends
            assert next_start < curr_end

    def test_short_text(self):
        text = "Short text."
        chunks = chunk_text(text, "test.txt", chunk_size=800, overlap=100)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_empty_text(self):
        assert chunk_text("", "test.txt") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\t  ", "test.txt") == []

    def test_chunk_id_format(self):
        text = _make_text(2000)
        chunks = chunk_text(text, "test.txt", chunk_size=800, overlap=100, file_index=3)
        for chunk in chunks:
            assert re.match(r"^doc\d{4}_chunk\d{4}$", chunk.chunk_id)
        assert chunks[0].chunk_id == "doc0003_chunk0000"
        assert chunks[1].chunk_id == "doc0003_chunk0001"

    def test_overlap_validation(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("some text", "test.txt", chunk_size=100, overlap=100)
        with pytest.raises(ValueError, match="overlap"):
            chunk_text("some text", "test.txt", chunk_size=100, overlap=200)

    def test_determinism(self):
        text = _make_text(2000)
        results = [
            chunk_text(text, "test.txt", chunk_size=800, overlap=100) for _ in range(10)
        ]
        first = results[0]
        for result in results[1:]:
            assert len(result) == len(first)
            for a, b in zip(result, first, strict=True):
                assert a.chunk_id == b.chunk_id
                assert a.text == b.text
                assert a.start_char == b.start_char
                assert a.end_char == b.end_char


class TestChunkCorpus:
    def test_corpus_directory(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("Hello world. " * 200, encoding="utf-8")
        (tmp_path / "b.txt").write_text("Foo bar baz. " * 200, encoding="utf-8")
        chunks = chunk_corpus(tmp_path, chunk_size=800, overlap=100)
        assert len(chunks) > 0
        sources = {c.source_file for c in chunks}
        assert sources == {"a.txt", "b.txt"}

    def test_corpus_sorted(self, tmp_path: Path):
        (tmp_path / "z_last.txt").write_text("zzz " * 300, encoding="utf-8")
        (tmp_path / "a_first.txt").write_text("aaa " * 300, encoding="utf-8")
        chunks = chunk_corpus(tmp_path, chunk_size=800, overlap=100)
        # First chunks should come from "a_first.txt" (sorted first)
        assert chunks[0].source_file == "a_first.txt"
        # File index 0 for sorted-first file
        assert chunks[0].chunk_id.startswith("doc0000_")

    def test_corpus_ignores_non_text(self, tmp_path: Path):
        (tmp_path / "script.py").write_text("print('hello')", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match=r"no \.txt or \.md files"):
            chunk_corpus(tmp_path)

    def test_corpus_nonexistent_dir(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not exist"):
            chunk_corpus(tmp_path / "missing")

    def test_corpus_reads_md_files(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text(
            "# Title\n\nContent here. " * 200, encoding="utf-8"
        )
        chunks = chunk_corpus(tmp_path)
        assert len(chunks) > 0
        assert chunks[0].source_file == "readme.md"
