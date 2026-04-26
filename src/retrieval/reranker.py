from __future__ import annotations

from sentence_transformers import CrossEncoder

from .vector_store import SearchResult


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)

        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        return reranked[:top_k] if top_k else reranked
