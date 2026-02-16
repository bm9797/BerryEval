"""BerryEval CLI interface."""

import sys


def app() -> None:
    """CLI entry point placeholder."""
    print(f"berryeval v{_get_version()}")
    sys.exit(0)


def _get_version() -> str:
    from berryeval import __version__

    return __version__
