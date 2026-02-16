"""Compare command implementation."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - required at runtime by Typer
from typing import TYPE_CHECKING, Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from berryeval.cli._app import app, state
from berryeval.cli._output import console, output_error, output_result
from berryeval.comparison.engine import compare_runs
from berryeval.comparison.thresholds import check_thresholds, parse_thresholds
from berryeval.persistence.results import load_result

if TYPE_CHECKING:
    from berryeval.comparison.types import ComparisonResult, ThresholdResult


def _display_comparison_human(
    comparison: ComparisonResult,
    threshold_results: list[ThresholdResult] | None,
) -> None:
    """Display comparison output in Rich tables."""
    console.print(
        Panel(
            (
                "[bold]Run Comparison[/bold]\n"
                f"Baseline:  {comparison.baseline_run_id[:8]}... "
                f"({comparison.baseline_num_queries} queries)\n"
                f"Candidate: {comparison.candidate_run_id[:8]}... "
                f"({comparison.candidate_num_queries} queries)"
            ),
            title="BerryEval Compare",
        )
    )

    for warning in comparison.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    metric_table = Table(title="Metric Comparison")
    metric_table.add_column("Metric", style="cyan")
    metric_table.add_column("Baseline", justify="right")
    metric_table.add_column("Candidate", justify="right")
    metric_table.add_column("Delta", justify="right")
    metric_table.add_column("Change %", justify="right")
    metric_table.add_column("Status")

    for delta in comparison.metric_deltas:
        direction = delta.direction.upper()
        if direction == "IMPROVED":
            status = f"[green]{direction}[/green]"
        elif direction == "REGRESSED":
            status = f"[red]{direction}[/red]"
        else:
            status = f"[yellow]{direction}[/yellow]"

        metric_table.add_row(
            delta.metric_name,
            f"{delta.baseline_value:.4f}",
            f"{delta.candidate_value:.4f}",
            f"{delta.absolute_delta:+.4f}",
            f"{delta.relative_delta_pct:+.2f}%",
            status,
        )

    console.print(metric_table)

    if threshold_results is not None:
        threshold_table = Table(title="Threshold Check")
        threshold_table.add_column("Metric", style="cyan")
        threshold_table.add_column("Threshold", justify="right")
        threshold_table.add_column("Actual", justify="right")
        threshold_table.add_column("Result")

        all_passed = True
        for tr in threshold_results:
            if tr.passed:
                result_str = "[green]PASS[/green]"
            else:
                result_str = "[red]FAIL[/red]"
                all_passed = False

            threshold_table.add_row(
                tr.metric_name,
                f"{tr.threshold:.4f}",
                f"{tr.actual:.4f}",
                result_str,
            )

        console.print(threshold_table)
        if all_passed:
            console.print("[green]All thresholds passed.[/green]")
        else:
            console.print("[red]Some thresholds failed.[/red]")

    if comparison.query_regressions:
        regression_table = Table(title="Per-Query Regressions")
        regression_table.add_column("Query", style="white", max_width=50)
        regression_table.add_column("Metric", style="cyan")
        regression_table.add_column("Baseline", justify="right")
        regression_table.add_column("Candidate", justify="right")
        regression_table.add_column("Delta", justify="right")

        for reg in comparison.query_regressions:
            regression_table.add_row(
                reg.query[:50],
                reg.metric_name,
                f"{reg.baseline_score:.4f}",
                f"{reg.candidate_score:.4f}",
                f"[red]{reg.delta:+.4f}[/red]",
            )

        console.print(regression_table)


@app.command()
def compare(
    baseline: Annotated[
        Path, typer.Argument(help="Path to baseline evaluation result JSON")
    ],
    candidate: Annotated[
        Path, typer.Argument(help="Path to candidate evaluation result JSON")
    ],
    fail_below: Annotated[
        str | None,
        typer.Option(
            "--fail-below",
            help=(
                "Threshold string, e.g. 'recall@10=0.80,precision@5=0.60'. "
                "Exit 1 if any metric fails."
            ),
        ),
    ] = None,
    per_query: Annotated[
        bool, typer.Option("--per-query", help="Show per-query regression details")
    ] = False,
) -> None:
    """Compare two evaluation runs and show metric deltas."""
    try:
        baseline_result = load_result(baseline)
    except FileNotFoundError:
        output_error(f"Baseline file not found: {baseline}")
        return
    except Exception as exc:
        output_error(f"Failed to load baseline: {exc}")
        return

    try:
        candidate_result = load_result(candidate)
    except FileNotFoundError:
        output_error(f"Candidate file not found: {candidate}")
        return
    except Exception as exc:
        output_error(f"Failed to load candidate: {exc}")
        return

    comparison_result = compare_runs(
        baseline_result, candidate_result, per_query=per_query
    )

    threshold_results: list[ThresholdResult] | None = None
    if fail_below is not None:
        try:
            thresholds = parse_thresholds(fail_below)
        except ValueError as exc:
            output_error(f"Invalid threshold format: {exc}")
            return
        threshold_results = check_thresholds(candidate_result, thresholds)

    if state.json_mode:
        payload: dict[str, object] = comparison_result.model_dump(mode="python")
        if threshold_results is not None:
            payload["threshold_results"] = [
                tr.model_dump(mode="python") for tr in threshold_results
            ]
            payload["thresholds_passed"] = all(tr.passed for tr in threshold_results)
        output_result(payload)
    else:
        _display_comparison_human(comparison_result, threshold_results)

    if threshold_results is not None and not all(tr.passed for tr in threshold_results):
        raise typer.Exit(code=1)
