"""Core Typer application and global state."""

import typer

app = typer.Typer(
    name="berryeval",
    help="Evaluate and benchmark RAG retrieval quality.",
    no_args_is_help=True,
)


class OutputState:
    json_mode: bool = False


state = OutputState()


@app.callback()
def main(
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON instead of formatted text"
    ),
) -> None:
    """BerryEval: RAG retrieval evaluation framework."""
    state.json_mode = json_output
