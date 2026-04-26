from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.hybrid_search import HybridSearcher, RetrievalMode, _reciprocal_rank_fusion
from src.retrieval.vector_store import SearchResult


def _make_result(id: int, score: float, content: str = "text") -> SearchResult:
    return SearchResult(id=id, score=score, content=content, metadata={})


class TestRRF:
    def test_merges_two_lists(self):
        a = [_make_result(1, 0.9), _make_result(2, 0.8), _make_result(3, 0.7)]
        b = [_make_result(2, 0.9), _make_result(1, 0.8), _make_result(4, 0.7)]
        fused = _reciprocal_rank_fusion([a, b])
        ids = [r.id for r in fused]
        # id 1 and 2 appear in both lists — should rank above 3 and 4
        assert ids.index(1) < ids.index(3)
        assert ids.index(2) < ids.index(4)

    def test_deduplicates(self):
        a = [_make_result(1, 0.9), _make_result(2, 0.8)]
        b = [_make_result(1, 0.9), _make_result(3, 0.7)]
        fused = _reciprocal_rank_fusion([a, b])
        assert len([r for r in fused if r.id == 1]) == 1

    def test_scores_assigned(self):
        a = [_make_result(1, 0.9)]
        fused = _reciprocal_rank_fusion([a])
        assert fused[0].score > 0

    def test_empty_lists(self):
        assert _reciprocal_rank_fusion([[], []]) == []

    def test_k_parameter_shifts_scores(self):
        a = [_make_result(1, 1.0)]
        low_k = _reciprocal_rank_fusion([a], k=1)
        high_k = _reciprocal_rank_fusion([a], k=1000)
        assert low_k[0].score > high_k[0].score


class TestHybridSearcher:
    def _mock_store(self, dense_results=None, corpus=None):
        store = MagicMock()
        store.search_dense.return_value = dense_results or []
        store.scroll_all.return_value = corpus or []
        return store

    def _mock_embedder(self):
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 1024
        return embedder

    def test_dense_mode_calls_store(self):
        results = [_make_result(1, 0.9)]
        store = self._mock_store(dense_results=results)
        searcher = HybridSearcher(store=store, embedder=self._mock_embedder())
        out = searcher.search("query", mode=RetrievalMode.DENSE, top_k=5)
        store.search_dense.assert_called_once()
        assert out == results

    def test_sparse_mode_uses_bm25(self):
        corpus = [
            _make_result(1, 0.0, content="machine learning transformer attention"),
            _make_result(2, 0.0, content="protein folding structure biology"),
        ]
        store = self._mock_store(corpus=corpus)
        searcher = HybridSearcher(store=store, embedder=self._mock_embedder())
        out = searcher.search("transformer attention", mode=RetrievalMode.SPARSE, top_k=2)
        assert len(out) > 0
        assert out[0].id == 1

    def test_hybrid_fuses_results(self):
        dense = [_make_result(1, 0.9), _make_result(2, 0.8)]
        corpus = [
            _make_result(1, 0.0, content="machine learning"),
            _make_result(2, 0.0, content="deep learning"),
            _make_result(3, 0.0, content="neural networks"),
        ]
        store = self._mock_store(dense_results=dense, corpus=corpus)
        searcher = HybridSearcher(store=store, embedder=self._mock_embedder())
        out = searcher.search("machine learning", mode=RetrievalMode.HYBRID, top_k=5)
        assert len(out) > 0

    def test_hybrid_rerank_calls_reranker(self):
        dense = [_make_result(1, 0.9)]
        corpus = [_make_result(1, 0.0, content="relevant text")]
        store = self._mock_store(dense_results=dense, corpus=corpus)
        reranker = MagicMock()
        reranker.rerank.return_value = dense
        searcher = HybridSearcher(store=store, embedder=self._mock_embedder())
        searcher.search("query", mode=RetrievalMode.HYBRID_RERANK, top_k=5, reranker=reranker)
        reranker.rerank.assert_called_once()

    def test_filters_passed_to_store(self):
        store = self._mock_store()
        searcher = HybridSearcher(store=store, embedder=self._mock_embedder())
        searcher.search("query", mode=RetrievalMode.DENSE, filters={"section": "results"})
        _, kwargs = store.search_dense.call_args
        assert kwargs.get("filters") == {"section": "results"}


class TestReranker:
    def test_reranks_by_score(self):
        from src.retrieval.reranker import Reranker
        reranker = MagicMock(spec=Reranker)
        results = [_make_result(1, 0.5, "relevant content"), _make_result(2, 0.9, "unrelated")]
        reranker.rerank.return_value = [results[0], results[1]]
        out = reranker.rerank("query", results, top_k=2)
        assert len(out) == 2
