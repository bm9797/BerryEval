"""Tests for berryeval.dataset.generator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from berryeval.config.types import Chunk
from berryeval.dataset.generator import (
    DEFAULT_PROMPT_TEMPLATE,
    generate_queries,
    generate_query_for_chunk,
)


def _make_mock_response(content: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_chunk(
    chunk_id: str = "doc0000_chunk0000", text: str = "Sample text."
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_file="test.txt",
        start_char=0,
        end_char=len(text),
    )


class TestGenerateQueryForChunk:
    def test_generate_single_query(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "test query"
        )

        result = generate_query_for_chunk("Some passage", client=mock_client)
        assert result == "test query"

    def test_prompt_template_used(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("query")
        custom_prompt = "Generate a question from this text."

        generate_query_for_chunk(
            "text", client=mock_client, prompt_template=custom_prompt
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["content"] == custom_prompt

    def test_model_parameter(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("query")

        generate_query_for_chunk("text", model="gpt-3.5-turbo", client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-3.5-turbo"

    def test_api_error_handling(self):
        from openai import OpenAIError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = OpenAIError("API failure")

        with pytest.raises(OpenAIError):
            generate_query_for_chunk("text", client=mock_client)

    def test_strips_whitespace(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "  query with spaces  \n"
        )

        result = generate_query_for_chunk("text", client=mock_client)
        assert result == "query with spaces"


class TestGenerateQueries:
    @patch("berryeval.dataset.generator.OpenAI")
    def test_generate_queries_multiple(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _make_mock_response("query 1"),
            _make_mock_response("query 2"),
            _make_mock_response("query 3"),
        ]

        chunks = [_make_chunk(f"doc0000_chunk{i:04d}", f"Text {i}") for i in range(3)]
        records = generate_queries(chunks)

        assert len(records) == 3
        assert records[0].query == "query 1"
        assert records[1].query == "query 2"
        assert records[2].query == "query 3"
        for i, rec in enumerate(records):
            assert rec.relevant_chunk_ids == [f"doc0000_chunk{i:04d}"]

    @patch("berryeval.dataset.generator.OpenAI")
    def test_progress_callback(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("q")

        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(3)]
        progress_calls: list[int] = []
        generate_queries(chunks, on_progress=lambda i: progress_calls.append(i))

        assert progress_calls == [0, 1, 2]

    @patch("berryeval.dataset.generator.OpenAI")
    def test_api_error_skips_chunk(self, mock_openai_cls):
        from openai import OpenAIError

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _make_mock_response("query 1"),
            OpenAIError("fail"),
            _make_mock_response("query 3"),
        ]

        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(3)]
        records = generate_queries(chunks)

        assert len(records) == 2
        assert records[0].query == "query 1"
        assert records[1].query == "query 3"

    @patch("berryeval.dataset.generator.OpenAI")
    def test_record_metadata(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("q")

        chunk = _make_chunk("doc0001_chunk0003", "Some text")
        records = generate_queries([chunk])

        assert len(records) == 1
        assert records[0].metadata["source_file"] == "test.txt"
        assert records[0].metadata["chunk_id"] == "doc0001_chunk0003"
        assert records[0].chunk_text == "Some text"


class TestDefaultPromptTemplate:
    def test_default_prompt_template_is_nonempty(self):
        assert len(DEFAULT_PROMPT_TEMPLATE) > 0

    def test_default_prompt_template_is_string(self):
        assert isinstance(DEFAULT_PROMPT_TEMPLATE, str)
