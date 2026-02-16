"""Tests for retriever adapter base types and registry."""

from __future__ import annotations

from abc import ABC

import pytest

from berryeval.retrievers.base import (
    _REGISTRY,
    RetrievedDocument,
    RetrieverAdapter,
    get_adapter_class,
    list_adapters,
    register_adapter,
)


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    """Restore adapter registry after each test."""
    original = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(original)


class TestRetrievedDocument:
    def test_construction(self):
        doc = RetrievedDocument(doc_id="doc-1", score=0.8, metadata={"source": "a"})
        assert doc.doc_id == "doc-1"
        assert doc.score == 0.8
        assert doc.metadata == {"source": "a"}

    def test_default_metadata(self):
        doc = RetrievedDocument(doc_id="doc-1", score=1.0)
        assert doc.metadata == {}

    def test_negative_score_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RetrievedDocument(doc_id="doc-1", score=-0.01)


class TestRetrieverAdapter:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            RetrieverAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class MockAdapter(RetrieverAdapter):
            name = "mock"

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return [RetrievedDocument(doc_id=f"{query}-{top_k}", score=1.0)]

            def close(self) -> None:
                return

        adapter = MockAdapter()
        results = adapter.retrieve("q", 3)
        assert len(results) == 1
        assert results[0].doc_id == "q-3"

    def test_context_manager_calls_close(self):
        class MockAdapter(RetrieverAdapter):
            name = "ctx"

            def __init__(self) -> None:
                self.closed = False

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return []

            def close(self) -> None:
                self.closed = True

        adapter = MockAdapter()
        with adapter as ctx_adapter:
            assert ctx_adapter.closed is False
        assert adapter.closed is True


class TestAdapterRegistry:
    def test_register_and_lookup(self):
        @register_adapter
        class AlphaAdapter(RetrieverAdapter):
            name = "alpha"

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return []

            def close(self) -> None:
                return

        adapter_cls = get_adapter_class("alpha")
        assert adapter_cls is AlphaAdapter

    def test_duplicate_name_raises(self):
        @register_adapter
        class AlphaAdapter(RetrieverAdapter):
            name = "alpha"

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return []

            def close(self) -> None:
                return

        with pytest.raises(ValueError, match="already registered"):

            @register_adapter
            class AlphaAdapterDuplicate(RetrieverAdapter):
                name = "alpha"

                def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                    return []

                def close(self) -> None:
                    return

    def test_unknown_name_raises_with_available(self):
        @register_adapter
        class OneAdapter(RetrieverAdapter):
            name = "one"

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return []

            def close(self) -> None:
                return

        @register_adapter
        class TwoAdapter(RetrieverAdapter):
            name = "two"

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return []

            def close(self) -> None:
                return

        with pytest.raises(KeyError) as excinfo:
            get_adapter_class("missing")

        message = str(excinfo.value)
        assert "Available adapters" in message
        assert "one" in message
        assert "two" in message

    def test_list_adapters_sorted(self):
        @register_adapter
        class ZetaAdapter(RetrieverAdapter):
            name = "zeta"

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return []

            def close(self) -> None:
                return

        @register_adapter
        class AlphaAdapter(RetrieverAdapter):
            name = "alpha"

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return []

            def close(self) -> None:
                return

        assert list_adapters() == ["alpha", "zeta"]

    def test_register_requires_name(self):
        class NamelessAdapter(RetrieverAdapter, ABC):
            name = ""

            def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
                return []

            def close(self) -> None:
                return

        with pytest.raises(ValueError, match="non-empty name"):
            register_adapter(NamelessAdapter)
