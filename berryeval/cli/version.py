"""Version command: display BerryEval and environment info."""

from __future__ import annotations

import sys

import numpy
from rich.table import Table

import berryeval
from berryeval.cli._app import app
from berryeval.cli._output import console, output_result
from berryeval.metrics import BACKEND


def _human_version(data: dict[str, object]) -> None:
    """Render version info as a Rich table."""
    table = Table(title="BerryEval Environment")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def version() -> None:
    """Show version and environment information."""
    data: dict[str, object] = {
        "berryeval": berryeval.__version__,
        "python": sys.version,
        "metrics_backend": BACKEND,
        "numpy": numpy.__version__,
    }
    output_result(data, human_fn=_human_version)
