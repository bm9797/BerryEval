"""Inspect command: examine dataset contents and metadata."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - required at runtime by Typer
from typing import Annotated

import typer
from rich.table import Table

from berryeval.cli._app import app
from berryeval.cli._output import console, output_error, output_result
from berryeval.dataset.reader import count_records, read_dataset_metadata


def _human_inspect(data: dict[str, object]) -> None:
    """Render dataset inspection as a Rich table."""
    table = Table(title="Dataset Inspection")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    config_hash = str(data.get("config_hash", ""))
    truncated_hash = config_hash[:16] + "..." if len(config_hash) > 16 else config_hash
    table.add_row("Config Hash", truncated_hash)
    table.add_row("Model", str(data.get("model", "")))
    table.add_row("Chunk Size", str(data.get("chunk_size", "")))
    table.add_row("Overlap", str(data.get("overlap", "")))
    table.add_row("Timestamp", str(data.get("timestamp", "")))
    table.add_row("Version", str(data.get("version", "")))
    table.add_row("Query Count", str(data.get("query_count", "")))

    corpus_stats = data.get("corpus_stats")
    if isinstance(corpus_stats, dict):
        files_count = corpus_stats.get("files", "N/A")
    else:
        files_count = "N/A"
    table.add_row("Corpus Files", str(files_count))

    console.print(table)
    console.print(f"\nFull config hash: {config_hash}")


@app.command(name="inspect")
def inspect_dataset(
    filepath: Annotated[
        Path, typer.Argument(help="Path to dataset JSONL file to inspect")
    ],
) -> None:
    """Examine dataset contents and metadata."""
    if not filepath.exists() or not filepath.is_file():
        output_error(f"File not found: {filepath}", code=1)

    try:
        metadata = read_dataset_metadata(filepath)
    except (ValueError, KeyError):
        output_error(
            "Invalid dataset: file does not contain valid BerryEval metadata",
            code=1,
        )
        return  # unreachable, but satisfies type checker
    except PermissionError:
        output_error(f"Permission denied: {filepath}", code=2)
        return

    query_count = count_records(filepath)

    data: dict[str, object] = {
        "config_hash": metadata.config_hash,
        "model": metadata.model,
        "chunk_size": metadata.chunk_size,
        "overlap": metadata.overlap,
        "timestamp": metadata.timestamp,
        "version": metadata.version,
        "prompt_template": metadata.prompt_template,
        "corpus_stats": metadata.corpus_stats,
        "query_count": query_count,
    }

    output_result(data, human_fn=_human_inspect)
