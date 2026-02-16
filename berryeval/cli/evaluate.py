"""Evaluate command stub (Phase 3)."""

from berryeval.cli._app import app
from berryeval.cli._output import output_error


@app.command()
def evaluate() -> None:
    """Run retrieval evaluation against a dataset."""
    output_error("Not yet implemented. Coming in Phase 3.")
