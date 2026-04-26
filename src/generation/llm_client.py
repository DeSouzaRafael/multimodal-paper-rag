from __future__ import annotations

import os
import re
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from ..retrieval.vector_store import SearchResult

load_dotenv()

_SYSTEM_PROMPT = """\
You are a research assistant that answers questions about scientific papers.

Rules:
- Answer ONLY using the provided context chunks.
- Cite every claim with [N] where N is the chunk number (1-indexed).
- If the context does not contain enough information to answer, say so — do not fabricate.
- When a chunk is a table, refer to it as a table and describe its contents accurately.
- When a chunk is a figure caption, describe what the figure shows based on the caption.
- Keep answers concise and precise."""

_CONTEXT_TEMPLATE = "[{n}] ({type} | {paper} p.{page} §{section})\n{content}"


@dataclass
class GenerationResult:
    answer: str
    citations: list[int]
    chunks_used: list[SearchResult]


def _format_context(chunks: list[SearchResult]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        parts.append(_CONTEXT_TEMPLATE.format(
            n=i,
            type=meta.get("element_type", "text"),
            paper=meta.get("paper_id", "unknown"),
            page=meta.get("page", "?"),
            section=meta.get("section", "?"),
            content=chunk.content[:800],
        ))
    return "\n\n".join(parts)


def _extract_citations(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})


class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._model = model

    def generate(
        self,
        query: str,
        chunks: list[SearchResult],
        max_tokens: int = 1024,
    ) -> GenerationResult:
        context = _format_context(chunks)
        user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        citations = _extract_citations(answer)

        return GenerationResult(
            answer=answer,
            citations=citations,
            chunks_used=chunks,
        )
