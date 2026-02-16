"""C extension wrapper for accelerated metric computation.

This module attempts to import the compiled C extension.
If unavailable, berryeval/metrics/__init__.py falls back to pure Python.

The C extension will be implemented in Phase 5.
"""

raise ImportError("C extension not available — using pure Python fallback")
