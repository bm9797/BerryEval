"""Text chunking utilities for corpus processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from berryeval.config.types import Chunk

if TYPE_CHECKING:
    from pathlib import Path


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 800,
    overlap: int = 100,
    file_index: int = 0,
) -> list[Chunk]:
    """Split text into overlapping chunks at word boundaries.

    Args:
        text: The text to chunk.
        source_file: Filename the text was read from.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between adjacent chunks.
        file_index: Index used for generating chunk IDs.

    Returns:
        A list of Chunk objects.

    Raises:
        ValueError: If overlap >= chunk_size.
    """
    if overlap >= chunk_size:
        msg = f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        raise ValueError(msg)

    if not text or text.isspace():
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_num = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Adjust to nearest word boundary
            space_pos = text.rfind(" ", start, end)
            if space_pos > start:
                end = space_pos
        else:
            end = len(text)

        chunk_text_str = text[start:end]

        if chunk_text_str.strip():
            chunk_id = f"doc{file_index:04d}_chunk{chunk_num:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    source_file=source_file,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_num += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def chunk_corpus(
    corpus_dir: Path,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """Read and chunk all .txt and .md files from a directory.

    Args:
        corpus_dir: Directory containing text files.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        A flat list of Chunk objects from all files.

    Raises:
        ValueError: If corpus_dir doesn't exist or contains no readable files.
    """
    if not corpus_dir.exists():
        msg = f"corpus directory does not exist: {corpus_dir}"
        raise ValueError(msg)

    files = sorted(p for p in corpus_dir.iterdir() if p.suffix in {".txt", ".md"})

    if not files:
        msg = f"no .txt or .md files found in: {corpus_dir}"
        raise ValueError(msg)

    all_chunks: list[Chunk] = []
    for file_index, filepath in enumerate(files):
        text = filepath.read_text(encoding="utf-8")
        file_chunks = chunk_text(
            text,
            source_file=filepath.name,
            chunk_size=chunk_size,
            overlap=overlap,
            file_index=file_index,
        )
        all_chunks.extend(file_chunks)

    return all_chunks
