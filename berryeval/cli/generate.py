"""Generate command: create evaluation datasets from a text corpus."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from berryeval.cli._app import app
from berryeval.cli._output import console, error_console, output_error, output_result
from berryeval.config.types import DatasetMetadata
from berryeval.dataset.chunker import chunk_corpus
from berryeval.dataset.generator import DEFAULT_PROMPT_TEMPLATE, generate_queries
from berryeval.dataset.hasher import compute_config_hash, compute_file_hash
from berryeval.dataset.writer import write_dataset


def _human_summary(data: dict[str, object]) -> None:
    """Render generation summary as a Rich table."""
    table = Table(title="Dataset Generated")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def generate(
    corpus: Annotated[
        Path, typer.Option("--corpus", help="Directory containing corpus documents")
    ],
    chunk_size: Annotated[
        int, typer.Option("--chunk-size", help="Characters per chunk")
    ] = 800,
    overlap: Annotated[
        int, typer.Option("--overlap", help="Character overlap between chunks")
    ] = 100,
    model: Annotated[
        str, typer.Option("--model", help="LLM model for query generation")
    ] = "gpt-4",
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output JSONL file path")
    ] = Path("dataset.jsonl"),
) -> None:
    """Generate an evaluation dataset from a text corpus using an LLM."""
    # --- Validate inputs ---
    if not corpus.is_dir():
        output_error(f"Corpus directory not found: {corpus}")

    if chunk_size <= 0:
        output_error("chunk-size must be a positive integer")

    if overlap >= chunk_size:
        output_error(f"overlap ({overlap}) must be less than chunk-size ({chunk_size})")

    output_dir = output.parent
    if not output_dir.exists():
        output_error(f"Output directory does not exist: {output_dir}")

    if not os.environ.get("OPENAI_API_KEY"):
        output_error(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it with: export OPENAI_API_KEY=your-key"
        )

    # --- Ingest corpus ---
    try:
        chunks = chunk_corpus(corpus, chunk_size=chunk_size, overlap=overlap)
    except ValueError as exc:
        output_error(str(exc))

    if not chunks:
        output_error(f"No text chunks produced from corpus: {corpus}")

    # --- Compute config hash ---
    corpus_files = sorted(corpus.iterdir())
    file_hashes = {
        f.name: compute_file_hash(f)
        for f in corpus_files
        if f.suffix in {".txt", ".md"}
    }
    config: dict[str, object] = {
        "file_hashes": file_hashes,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model": model,
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
    }
    config_hash = compute_config_hash(config)

    # --- Generate queries ---
    error_console.print(
        f"[bold]Generating queries for {len(chunks)} chunks with {model}...[/bold]"
    )
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=error_console,
        ) as progress:
            task = progress.add_task("Generating queries...", total=len(chunks))

            def on_progress(_i: int) -> None:
                progress.update(task, advance=1)

            records = generate_queries(
                chunks,
                model=model,
                prompt_template=DEFAULT_PROMPT_TEMPLATE,
                on_progress=on_progress,
            )
    except Exception as exc:
        output_error(f"Query generation failed: {exc}", code=2)
        return  # unreachable, but satisfies type checker

    if not records:
        output_error(
            "No queries were generated. Check API key and model access.", code=2
        )

    # --- Build metadata ---
    source_files = sorted({c.source_file for c in chunks})
    corpus_stats: dict[str, object] = {
        "total_files": len(source_files),
        "total_chunks": len(chunks),
        "source_files": source_files,
    }
    metadata = DatasetMetadata(
        config_hash=config_hash,
        model=model,
        chunk_size=chunk_size,
        overlap=overlap,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        corpus_stats=corpus_stats,
        version="0.1.0",
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    )

    # --- Write dataset ---
    write_dataset(output, metadata, records)

    # --- Output summary ---
    data: dict[str, object] = {
        "output_file": str(output),
        "config_hash": config_hash,
        "query_count": len(records),
        "corpus_stats": corpus_stats,
        "model": model,
    }
    output_result(data, human_fn=_human_summary)
