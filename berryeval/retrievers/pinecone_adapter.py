"""Pinecone retriever adapter."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from berryeval.retrievers.base import (
    RetrievedDocument,
    RetrieverAdapter,
    register_adapter,
)

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None


@register_adapter
class PineconeAdapter(RetrieverAdapter):
    """Retriever adapter backed by a Pinecone index."""

    name = "pinecone"

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "",
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: str = "",
        **kwargs: object,
    ) -> None:
        del kwargs
        if Pinecone is None:
            msg = (
                "pinecone-client is required. "
                "Install with: pip install berryeval[pinecone]"
            )
            raise RuntimeError(msg)

        self._client: Any = Pinecone(api_key=api_key)
        self._index: Any = self._client.Index(index_name)
        self._namespace = namespace
        self._embedding_model = embedding_model
        self._openai = (
            OpenAI(api_key=embedding_api_key)
            if embedding_api_key
            else OpenAI()
        )

    def _embed(self, text: str) -> list[float]:
        response: Any = self._openai.embeddings.create(
            input=text,
            model=self._embedding_model,
        )
        embedding = response.data[0].embedding
        return [float(value) for value in embedding]

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        vector = self._embed(query)
        response: Any = self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=self._namespace,
            include_metadata=True,
        )

        results: list[RetrievedDocument] = []
        for match in response.matches:
            metadata = dict(match.metadata) if match.metadata else {}
            results.append(
                RetrievedDocument(
                    doc_id=str(match.id),
                    score=float(match.score),
                    metadata=metadata,
                )
            )
        return results

    def close(self) -> None:
        """No-op: pinecone client does not require explicit close."""
        return
