"""Evaluate command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.panel import Panel
from rich.table import Table

from berryeval.cli._app import app
from berryeval.cli._output import console, output_error, output_result
from berryeval.config.eval_config import load_eval_config
from berryeval.persistence.results import save_result
from berryeval.retrievers.base import get_adapter_class
from berryeval.runner.evaluator import EvaluationRunner

if TYPE_CHECKING:
    from berryeval.persistence.types import RunResult


def _display_human_results(result: RunResult, result_path: Path) -> None:
    """Display evaluation output in Rich tables."""
    console.print(
        Panel(
            (
                "[bold]Evaluation Complete[/bold]\n"
                f"Run ID: {result.run_id[:8]}...\n"
                f"Queries: {result.num_queries}\n"
                f"Dataset: {result.dataset_path}"
            ),
            title="BerryEval Results",
        )
    )

    metrics_table = Table(title="Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("k", justify="right")
    metrics_table.add_column("Score", justify="right", style="green")

    for metric in result.metrics:
        metric_name = metric.name.split("@")[0] if "@" in metric.name else metric.name
        metrics_table.add_row(metric_name, str(metric.k), f"{metric.mean:.4f}")

    console.print(metrics_table)

    latency_table = Table(title="Latency (ms)")
    latency_table.add_column("Statistic", style="cyan")
    latency_table.add_column("Value", justify="right", style="yellow")

    latency = result.latency
    latency_table.add_row("p50", f"{latency.p50:.1f}")
    latency_table.add_row("p95", f"{latency.p95:.1f}")
    latency_table.add_row("p99", f"{latency.p99:.1f}")
    latency_table.add_row("mean", f"{latency.mean:.1f}")
    latency_table.add_row("min", f"{latency.min_ms:.1f}")
    latency_table.add_row("max", f"{latency.max_ms:.1f}")

    console.print(latency_table)

    if result.query_breakdowns:
        metric_columns = sorted(result.query_breakdowns[0].scores)

        per_query_table = Table(title="Per-Query Breakdown")
        per_query_table.add_column("Query", style="white", max_width=50)
        per_query_table.add_column("Latency (ms)", justify="right")
        for metric_name in metric_columns:
            per_query_table.add_column(metric_name, justify="right")

        for breakdown in result.query_breakdowns:
            row = [breakdown.query[:50], f"{breakdown.latency_ms:.1f}"]
            for metric_name in metric_columns:
                row.append(f"{breakdown.scores[metric_name]:.4f}")
            per_query_table.add_row(*row)

        console.print(per_query_table)

    console.print(f"\n[dim]Results saved to: {result_path}[/dim]")


@app.command()
def evaluate(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to evaluation dataset JSONL file", exists=True),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to evaluation YAML config file", exists=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for results"),
    ] = Path("./results"),
    per_query: Annotated[
        bool, typer.Option("--per-query", help="Show per-query metric breakdowns")
    ] = False,
    k_values: Annotated[
        str | None,
        typer.Option("--k", help="Override k values (comma-separated, e.g. '5,10,20')"),
    ] = None,
) -> None:
    """Run retrieval evaluation against a dataset."""
    try:
        eval_config = load_eval_config(config)
    except (FileNotFoundError, ValueError, ValidationError, yaml.YAMLError) as exc:
        output_error(f"Config error: {exc}")
        return

    if k_values is not None:
        try:
            parsed = [int(value.strip()) for value in k_values.split(",") if value.strip()]
            if not parsed or any(value < 1 for value in parsed):
                raise ValueError
            eval_config.evaluation.k_values = parsed
        except ValueError:
            output_error("Invalid --k value. Use comma-separated positive integers")
            return

    eval_config.evaluation.per_query = per_query

    adapter = None
    try:
        adapter_cls = get_adapter_class(eval_config.retriever.type)
        adapter_kwargs = {
            key: value
            for key, value in eval_config.retriever.model_dump(mode="python").items()
            if key != "type"
        }
        adapter = adapter_cls(**adapter_kwargs)
    except Exception as exc:
        output_error(f"Retriever error: {exc}")
        return

    try:
        runner = EvaluationRunner(adapter, eval_config)
        result = runner.run(dataset)
    except Exception as exc:
        output_error(f"Evaluation failed: {exc}")
        return
    finally:
        adapter.close()

    result_path = save_result(result, output)

    payload = result.model_dump(mode="python")
    output_result(
        payload,
        human_fn=lambda _data: _display_human_results(result, result_path),
    )
