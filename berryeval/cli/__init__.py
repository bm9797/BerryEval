"""BerryEval CLI interface."""

import berryeval.cli.compare as compare
import berryeval.cli.evaluate as evaluate
import berryeval.cli.generate as generate
import berryeval.cli.inspect_cmd as inspect_cmd
import berryeval.cli.version as version
from berryeval.cli._app import app

__all__ = ["app", "compare", "evaluate", "generate", "inspect_cmd", "version"]
