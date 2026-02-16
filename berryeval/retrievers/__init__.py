"""Retriever adapter system."""

from contextlib import suppress

from berryeval.retrievers.base import (
    RetrievedDocument,
    RetrieverAdapter,
    get_adapter_class,
    list_adapters,
    register_adapter,
)

# Auto-register built-in adapters when optional dependencies are available.
with suppress(ImportError):
    import berryeval.retrievers.pinecone_adapter  # noqa: F401

__all__ = [
    "RetrievedDocument",
    "RetrieverAdapter",
    "get_adapter_class",
    "list_adapters",
    "register_adapter",
]
