"""Output helpers for JSON and human-readable modes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from berryeval.cli._app import state

if TYPE_CHECKING:
    from collections.abc import Callable

console = Console()
error_console = Console(stderr=True)


def output_result(
    data: dict[str, object],
    human_fn: Callable[[dict[str, object]], None] | None = None,
) -> None:
    """Print result in JSON or human-readable format."""
    if state.json_mode:
        console.print_json(json.dumps(data))
    elif human_fn is not None:
        human_fn(data)
    else:
        console.print_json(json.dumps(data))


def output_error(message: str, code: int = 1) -> None:
    """Print error and exit with given code."""
    if state.json_mode:
        console.print_json(json.dumps({"error": message}))
    else:
        error_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=code)
