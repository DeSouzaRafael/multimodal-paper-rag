import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.test_dataset import QAPair, _GEN_PROMPT, load_dataset
from src.evaluation.ragas_eval import _run_rag
from src.retrieval.hybrid_search import RetrievalMode
from src.retrieval.vector_store import SearchResult


def _make_pair(**kwargs) -> QAPair:
    defaults = dict(
        question="What method was used?",
        answer="They used cross-attention.",
        type="factual",
        chunk_id=1,
        paper_id="2310.00001",
        section="methods",
        page=3,
    )
    return QAPair(**{**defaults, **kwargs})


def _make_chunk(id: int, content: str = "some content") -> SearchResult:
    return SearchResult(
        id=id, score=0.9, content=content,
        metadata={"paper_id": "2310.00001", "page": 1, "section": "methods", "element_type": "text"},
    )


class TestLoadDataset:
    def test_roundtrip(self):
        pairs = [_make_pair(), _make_pair(question="Another question?", type="table")]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            import json, dataclasses
            json.dump([dataclasses.asdict(p) for p in pairs], f)
            path = f.name
        loaded = load_dataset(path)
        assert len(loaded) == 2
        assert loaded[0].question == "What method was used?"
        assert loaded[1].type == "table"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path.json")


class TestRunRag:
    def test_returns_parallel_lists(self):
        pairs = [_make_pair(), _make_pair(question="Second?")]
        chunks = [_make_chunk(1, "relevant")]

        searcher = MagicMock()
        searcher.search.return_value = chunks

        llm = MagicMock()
        llm.generate.return_value = MagicMock(answer="The answer is [1].")

        questions, answers, contexts = _run_rag(
            pairs, searcher, llm, RetrievalMode.DENSE, reranker=None
        )

        assert len(questions) == len(answers) == len(contexts) == 2
        assert questions[0] == pairs[0].question
        assert contexts[0] == [c.content for c in chunks]

    def test_reranker_passed_only_for_hybrid_rerank(self):
        pairs = [_make_pair()]
        searcher = MagicMock()
        searcher.search.return_value = [_make_chunk(1)]
        llm = MagicMock()
        llm.generate.return_value = MagicMock(answer="answer")
        reranker = MagicMock()

        _run_rag(pairs, searcher, llm, RetrievalMode.DENSE, reranker=reranker)
        _, kwargs = searcher.search.call_args
        assert kwargs.get("reranker") is None

        _run_rag(pairs, searcher, llm, RetrievalMode.HYBRID_RERANK, reranker=reranker)
        _, kwargs = searcher.search.call_args
        assert kwargs.get("reranker") is reranker


class TestQAPair:
    def test_valid_types(self):
        for t in ("factual", "aggregation", "table", "figure"):
            pair = _make_pair(type=t)
            assert pair.type == t

    def test_gen_prompt_has_placeholders(self):
        assert "{n}" in _GEN_PROMPT
        assert "{chunk}" in _GEN_PROMPT
