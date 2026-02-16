"""Data models for BerryEval configuration and dataset records."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single text chunk extracted from a source document."""

    chunk_id: str
    text: str
    source_file: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class DatasetMetadata:
    """Metadata header for a dataset file."""

    config_hash: str
    model: str
    chunk_size: int
    overlap: int
    timestamp: str
    corpus_stats: dict[str, object]
    version: str
    prompt_template: str

    def to_dict(self) -> dict[str, object]:
        return {
            "config_hash": self.config_hash,
            "model": self.model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "timestamp": self.timestamp,
            "corpus_stats": self.corpus_stats,
            "version": self.version,
            "prompt_template": self.prompt_template,
        }


@dataclass
class DatasetRecord:
    """A single query-chunk pair in a dataset."""

    query: str
    relevant_chunk_ids: list[str]
    chunk_text: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "chunk_text": self.chunk_text,
            "metadata": self.metadata,
        }
