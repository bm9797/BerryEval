"""Retriever adapter base types and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(slots=True)
class RetrievedDocument:
    """A single retrieved document with score and optional metadata."""

    doc_id: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score < 0:
            msg = "score must be non-negative"
            raise ValueError(msg)


class RetrieverAdapter(ABC):
    """Abstract adapter contract for retriever backends."""

    name: ClassVar[str]

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        """Return top-k retrieved documents for the query."""

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""

    def __enter__(self) -> RetrieverAdapter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


_REGISTRY: dict[str, type[RetrieverAdapter]] = {}


def register_adapter(adapter_cls: type[RetrieverAdapter]) -> type[RetrieverAdapter]:
    """Register an adapter class by its declared name."""
    name = getattr(adapter_cls, "name", "")
    if not name:
        msg = f"Adapter class {adapter_cls.__name__} must define a non-empty name"
        raise ValueError(msg)
    if name in _REGISTRY:
        msg = f"Adapter '{name}' is already registered"
        raise ValueError(msg)
    _REGISTRY[name] = adapter_cls
    return adapter_cls


def get_adapter_class(name: str) -> type[RetrieverAdapter]:
    """Look up a registered adapter class by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_adapters()) or "(none)"
        msg = f"Unknown adapter '{name}'. Available adapters: {available}"
        raise KeyError(msg) from exc


def list_adapters() -> list[str]:
    """Return all registered adapter names sorted alphabetically."""
    return sorted(_REGISTRY)
