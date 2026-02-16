"""LLM-powered query generation for dataset creation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from openai import OpenAI, OpenAIError

from berryeval.config.types import DatasetRecord

if TYPE_CHECKING:
    from collections.abc import Callable

    from berryeval.config.types import Chunk

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = (
    "You are a search query generator. Given a text passage, generate a "
    "realistic search query that a user might type to find this information. "
    "The query should be natural, concise, and specific to the content. "
    "Return ONLY the query text, with no additional explanation or formatting."
)


def generate_query_for_chunk(
    chunk_text: str,
    model: str = "gpt-4",
    client: OpenAI | None = None,
    prompt_template: str | None = None,
) -> str:
    """Generate a search query for a single text chunk using an LLM.

    Args:
        chunk_text: The text passage to generate a query for.
        model: The LLM model to use.
        client: An optional pre-configured OpenAI client.
        prompt_template: Custom system prompt; defaults to DEFAULT_PROMPT_TEMPLATE.

    Returns:
        The generated query string.

    Raises:
        OpenAIError: If the API call fails.
    """
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = prompt_template or DEFAULT_PROMPT_TEMPLATE

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text},
            ],
            temperature=0.7,
        )
    except OpenAIError as exc:
        msg = f"OpenAI API error: {exc}"
        raise OpenAIError(msg) from exc

    content = response.choices[0].message.content
    return (content or "").strip()


def generate_queries(
    chunks: list[Chunk],
    model: str = "gpt-4",
    prompt_template: str | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> list[DatasetRecord]:
    """Generate search queries for a list of text chunks.

    Creates a single OpenAI client and reuses it for all calls.

    Args:
        chunks: List of Chunk objects to generate queries for.
        model: The LLM model to use.
        prompt_template: Custom system prompt; defaults to DEFAULT_PROMPT_TEMPLATE.
        on_progress: Optional callback invoked after each chunk with the
            current index (0-based).

    Returns:
        A list of DatasetRecord objects, one per successfully processed chunk.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    records: list[DatasetRecord] = []

    for i, chunk in enumerate(chunks):
        try:
            query = generate_query_for_chunk(
                chunk.text,
                model=model,
                client=client,
                prompt_template=prompt_template,
            )
            records.append(
                DatasetRecord(
                    query=query,
                    relevant_chunk_ids=[chunk.chunk_id],
                    chunk_text=chunk.text,
                    metadata={
                        "source_file": chunk.source_file,
                        "chunk_id": chunk.chunk_id,
                    },
                )
            )
        except OpenAIError:
            logger.warning(
                "Failed to generate query for chunk %s, skipping", chunk.chunk_id
            )

        if on_progress is not None:
            on_progress(i)

    return records
