"""Compare command stub (Phase 4)."""

from berryeval.cli._app import app
from berryeval.cli._output import output_error


@app.command()
def compare() -> None:
    """Compare evaluation results across runs."""
    output_error("Not yet implemented. Coming in Phase 4.")
