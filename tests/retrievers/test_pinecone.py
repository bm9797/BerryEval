"""Tests for Pinecone retriever adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from berryeval.retrievers.base import get_adapter_class
from berryeval.retrievers.pinecone_adapter import PineconeAdapter


@pytest.fixture
def mock_pinecone(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    mock_client = MagicMock()
    mock_index = MagicMock()
    mock_client.Index.return_value = mock_index
    mock_pinecone_cls = MagicMock(return_value=mock_client)

    monkeypatch.setattr(
        "berryeval.retrievers.pinecone_adapter.Pinecone",
        mock_pinecone_cls,
    )
    return {
        "client_cls": mock_pinecone_cls,
        "client": mock_client,
        "index": mock_index,
    }


@pytest.fixture
def mock_openai(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    mock_client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client.embeddings.create.return_value = mock_embedding
    mock_openai_cls = MagicMock(return_value=mock_client)

    monkeypatch.setattr(
        "berryeval.retrievers.pinecone_adapter.OpenAI",
        mock_openai_cls,
    )
    return {
        "client_cls": mock_openai_cls,
        "client": mock_client,
    }


class TestPineconeAdapter:
    def test_adapter_registered(self):
        adapter_cls = get_adapter_class("pinecone")
        assert adapter_cls is PineconeAdapter

    def test_adapter_name(self):
        assert PineconeAdapter.name == "pinecone"

    def test_init_creates_clients(
        self,
        mock_pinecone: dict[str, Any],
        mock_openai: dict[str, Any],
    ):
        _adapter = PineconeAdapter(api_key="test-key", index_name="test-index")

        mock_pinecone["client_cls"].assert_called_once_with(api_key="test-key")
        mock_pinecone["client"].Index.assert_called_once_with("test-index")
        mock_openai["client_cls"].assert_called_once_with()

    def test_init_with_embedding_api_key(
        self,
        mock_pinecone: dict[str, Any],
        mock_openai: dict[str, Any],
    ):
        _adapter = PineconeAdapter(
            api_key="test-key",
            index_name="test-index",
            embedding_api_key="emb-key",
        )

        mock_openai["client_cls"].assert_called_once_with(api_key="emb-key")

    def test_init_without_pinecone_installed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr("berryeval.retrievers.pinecone_adapter.Pinecone", None)
        with pytest.raises(RuntimeError, match=r"pip install berryeval\[pinecone\]"):
            PineconeAdapter(api_key="test-key", index_name="test-index")

    def test_retrieve_embeds_and_queries(
        self,
        mock_pinecone: dict[str, Any],
        mock_openai: dict[str, Any],
    ):
        mock_match_1 = MagicMock()
        mock_match_1.id = "doc1"
        mock_match_1.score = 0.95
        mock_match_1.metadata = {"text": "hello"}

        mock_match_2 = MagicMock()
        mock_match_2.id = "doc2"
        mock_match_2.score = 0.81
        mock_match_2.metadata = {"text": "world"}

        mock_response = MagicMock()
        mock_response.matches = [mock_match_1, mock_match_2]
        mock_pinecone["index"].query.return_value = mock_response

        adapter = PineconeAdapter(api_key="test-key", index_name="test-index")
        results = adapter.retrieve("test query", top_k=2)

        mock_openai["client"].embeddings.create.assert_called_once_with(
            input="test query",
            model="text-embedding-3-small",
        )
        mock_pinecone["index"].query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=2,
            namespace="",
            include_metadata=True,
        )

        assert len(results) == 2
        assert results[0].doc_id == "doc1"
        assert results[0].score == 0.95
        assert results[0].metadata == {"text": "hello"}
        assert results[1].doc_id == "doc2"

    def test_retrieve_custom_embedding_model(
        self,
        mock_pinecone: dict[str, Any],
        mock_openai: dict[str, Any],
    ):
        mock_response = MagicMock()
        mock_response.matches = []
        mock_pinecone["index"].query.return_value = mock_response

        adapter = PineconeAdapter(
            api_key="test-key",
            index_name="test-index",
            embedding_model="text-embedding-ada-002",
        )
        adapter.retrieve("hello", top_k=1)

        mock_openai["client"].embeddings.create.assert_called_once_with(
            input="hello",
            model="text-embedding-ada-002",
        )

    def test_retrieve_empty_results(
        self,
        mock_pinecone: dict[str, Any],
        mock_openai: dict[str, Any],
    ):
        mock_response = MagicMock()
        mock_response.matches = []
        mock_pinecone["index"].query.return_value = mock_response

        adapter = PineconeAdapter(api_key="test-key", index_name="test-index")
        results = adapter.retrieve("test query", top_k=3)

        assert results == []

    def test_retrieve_with_namespace(
        self,
        mock_pinecone: dict[str, Any],
        mock_openai: dict[str, Any],
    ):
        mock_response = MagicMock()
        mock_response.matches = []
        mock_pinecone["index"].query.return_value = mock_response

        adapter = PineconeAdapter(
            api_key="test-key",
            index_name="test-index",
            namespace="my-ns",
        )
        adapter.retrieve("query", top_k=1)

        mock_pinecone["index"].query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3],
            top_k=1,
            namespace="my-ns",
            include_metadata=True,
        )

    def test_context_manager(
        self,
        mock_pinecone: dict[str, Any],
        mock_openai: dict[str, Any],
    ):
        with PineconeAdapter(api_key="test-key", index_name="test-index") as adapter:
            assert isinstance(adapter, PineconeAdapter)

    def test_close(
        self,
        mock_pinecone: dict[str, Any],
        mock_openai: dict[str, Any],
    ):
        adapter = PineconeAdapter(api_key="test-key", index_name="test-index")
        adapter.close()
