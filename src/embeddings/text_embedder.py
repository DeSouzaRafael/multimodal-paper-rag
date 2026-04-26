from __future__ import annotations

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
