from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "papers")


@dataclass
class SearchResult:
    id: str | int
    score: float
    content: str
    metadata: dict
    image_path: str | None = None


def _build_filter(filters: dict | None) -> Filter | None:
    if not filters:
        return None
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    return Filter(must=conditions)


class VectorStore:
    def __init__(self, host: str | None = None, port: int | None = None):
        self._client = QdrantClient(
            host=host or os.getenv("QDRANT_HOST", "localhost"),
            port=port or int(os.getenv("QDRANT_PORT", "6333")),
        )

    def search_dense(
        self,
        vector: list[float],
        vector_name: str = "text",
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        hits = self._client.search(
            collection_name=COLLECTION,
            query_vector=(vector_name, vector),
            limit=top_k,
            query_filter=_build_filter(filters),
            with_payload=True,
        )
        return [
            SearchResult(
                id=h.id,
                score=h.score,
                content=h.payload.get("content", ""),
                metadata={k: v for k, v in h.payload.items() if k not in ("content", "image_path")},
                image_path=h.payload.get("image_path"),
            )
            for h in hits
        ]

    def scroll_all(self, filters: dict | None = None) -> list[SearchResult]:
        results = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=COLLECTION,
                scroll_filter=_build_filter(filters),
                with_payload=True,
                limit=256,
                offset=offset,
            )
            for r in records:
                results.append(SearchResult(
                    id=r.id,
                    score=0.0,
                    content=r.payload.get("content", ""),
                    metadata={k: v for k, v in r.payload.items() if k not in ("content", "image_path")},
                    image_path=r.payload.get("image_path"),
                ))
            if offset is None:
                break
        return results
