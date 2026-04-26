from unittest.mock import MagicMock, patch

from src.generation.llm_client import (
    GenerationResult,
    LLMClient,
    _extract_citations,
    _format_context,
)
from src.retrieval.vector_store import SearchResult


def _chunk(id: int, content: str, page: int = 1, section: str = "intro") -> SearchResult:
    return SearchResult(
        id=id,
        score=0.9,
        content=content,
        metadata={"paper_id": "2310.00001", "page": page, "section": section, "element_type": "text"},
    )


class TestExtractCitations:
    def test_single(self):
        assert _extract_citations("See [1] for details.") == [1]

    def test_multiple(self):
        assert _extract_citations("As shown in [2] and [3], the [1] approach works.") == [1, 2, 3]

    def test_duplicates_deduped(self):
        assert _extract_citations("[1] blah [1] blah") == [1]

    def test_no_citations(self):
        assert _extract_citations("No citations here.") == []


class TestFormatContext:
    def test_numbered_correctly(self):
        chunks = [_chunk(1, "first"), _chunk(2, "second")]
        ctx = _format_context(chunks)
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_metadata_included(self):
        chunks = [_chunk(1, "content", page=5, section="results")]
        ctx = _format_context(chunks)
        assert "p.5" in ctx
        assert "results" in ctx

    def test_content_truncated_at_800(self):
        long_content = "word " * 300
        chunks = [_chunk(1, long_content)]
        ctx = _format_context(chunks)
        assert len(ctx) < len(long_content) + 500


class TestLLMClient:
    def _make_client(self, answer: str = "Based on [1] the result is X.") -> LLMClient:
        client = LLMClient.__new__(LLMClient)
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=answer)]
        mock_anthropic.messages.create.return_value = mock_response
        client._client = mock_anthropic
        client._model = "claude-sonnet-4-6"
        return client

    def test_returns_generation_result(self):
        client = self._make_client()
        chunks = [_chunk(1, "relevant content")]
        result = client.generate("What happened?", chunks)
        assert isinstance(result, GenerationResult)

    def test_answer_extracted(self):
        client = self._make_client(answer="The answer is [1].")
        chunks = [_chunk(1, "content")]
        result = client.generate("query", chunks)
        assert result.answer == "The answer is [1]."

    def test_citations_parsed(self):
        client = self._make_client(answer="See [2] and [1] for this.")
        chunks = [_chunk(1, "a"), _chunk(2, "b")]
        result = client.generate("query", chunks)
        assert result.citations == [1, 2]

    def test_chunks_preserved(self):
        client = self._make_client()
        chunks = [_chunk(1, "content")]
        result = client.generate("query", chunks)
        assert result.chunks_used == chunks

    def test_api_called_with_system_prompt(self):
        client = self._make_client()
        chunks = [_chunk(1, "content")]
        client.generate("query", chunks)
        call_kwargs = client._client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert "cite" in call_kwargs["system"].lower() or "citation" in call_kwargs["system"].lower()
