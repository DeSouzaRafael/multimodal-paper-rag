from __future__ import annotations

from collections import defaultdict
from enum import Enum

from rank_bm25 import BM25Okapi

from ..embeddings.text_embedder import TextEmbedder
from .vector_store import SearchResult, VectorStore


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid+rerank"


def _reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    scores: dict[str | int, float] = defaultdict(float)
    result_map: dict[str | int, SearchResult] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            scores[result.id] += 1.0 / (k + rank + 1)
            result_map[result.id] = result

    fused = sorted(result_map.values(), key=lambda r: scores[r.id], reverse=True)
    for r in fused:
        r.score = scores[r.id]
    return fused


class HybridSearcher:
    def __init__(self, store: VectorStore | None = None, embedder: TextEmbedder | None = None):
        self._store = store or VectorStore()
        self._embedder = embedder or TextEmbedder()
        self._bm25: BM25Okapi | None = None
        self._corpus: list[SearchResult] = []

    def _ensure_bm25(self, filters: dict | None = None) -> None:
        docs = self._store.scroll_all(filters=filters)
        if not docs:
            return
        self._corpus = docs
        tokenized = [d.content.lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenized)

    def _sparse_search(self, query: str, top_k: int, filters: dict | None) -> list[SearchResult]:
        self._ensure_bm25(filters)
        if not self._bm25 or not self._corpus:
            return []
        tokens = query.lower().split()
        raw_scores = self._bm25.get_scores(tokens)
        indexed = sorted(
            enumerate(raw_scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        results = []
        for idx, score in indexed:
            r = self._corpus[idx]
            r.score = float(score)
            results.append(r)
        return results

    def search(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 10,
        filters: dict | None = None,
        reranker=None,
    ) -> list[SearchResult]:
        vec = self._embedder.embed(query)

        if mode == RetrievalMode.DENSE:
            return self._store.search_dense(vec, top_k=top_k, filters=filters)

        if mode == RetrievalMode.SPARSE:
            return self._sparse_search(query, top_k=top_k, filters=filters)

        dense_results = self._store.search_dense(vec, top_k=top_k * 2, filters=filters)
        sparse_results = self._sparse_search(query, top_k=top_k * 2, filters=filters)
        fused = _reciprocal_rank_fusion([dense_results, sparse_results])[:top_k]

        if mode == RetrievalMode.HYBRID_RERANK and reranker is not None:
            fused = reranker.rerank(query, fused, top_k=top_k)

        return fused
