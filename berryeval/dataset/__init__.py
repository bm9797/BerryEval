"""Dataset generation and management."""

from berryeval.dataset.chunker import chunk_corpus, chunk_text
from berryeval.dataset.generator import generate_queries, generate_query_for_chunk
from berryeval.dataset.hasher import compute_config_hash, compute_file_hash
from berryeval.dataset.reader import (
    count_records,
    read_dataset_metadata,
    read_dataset_records,
)
from berryeval.dataset.writer import write_dataset

__all__ = [
    "chunk_corpus",
    "chunk_text",
    "compute_config_hash",
    "compute_file_hash",
    "count_records",
    "generate_queries",
    "generate_query_for_chunk",
    "read_dataset_metadata",
    "read_dataset_records",
    "write_dataset",
]
