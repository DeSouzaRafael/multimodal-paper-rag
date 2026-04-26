from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .parser import ElementType, ParsedElement


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    element_type: ElementType = ElementType.TEXT
    image_path: str | None = None


_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _fixed_split(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


def _recursive_split(text: str, size: int, overlap: int, separators: list[str]) -> list[str]:
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate.split()) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if len(part.split()) > size:
                        chunks.extend(_recursive_split(part, size, overlap, separators[separators.index(sep) + 1:] or [" "]))
                        current = ""
                    else:
                        current = part.strip()
            if current:
                chunks.append(current)
            return chunks if chunks else [text]
    return _fixed_split(text, size, overlap)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _semantic_split(sentences: list[str], model: SentenceTransformer, threshold: float = 0.5) -> list[list[str]]:
    if len(sentences) <= 1:
        return [sentences]

    embeddings = model.encode(sentences, normalize_embeddings=True)
    groups: list[list[str]] = [[sentences[0]]]

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            groups.append([])
        groups[-1].append(sentences[i])

    return groups


def chunk_elements(
    elements: list[ParsedElement],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = 300,
    overlap: int = 50,
    semantic_model: str = "BAAI/bge-m3",
    semantic_threshold: float = 0.5,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    _model: SentenceTransformer | None = None

    if strategy == ChunkStrategy.SEMANTIC:
        _model = SentenceTransformer(semantic_model)

    for el in elements:
        # Tables and images pass through as single chunks — don't split them
        if el.type in (ElementType.TABLE, ElementType.IMAGE):
            chunks.append(Chunk(
                content=el.content,
                metadata=el.metadata,
                element_type=el.type,
                image_path=el.image_path,
            ))
            continue

        text = el.content

        if strategy == ChunkStrategy.FIXED:
            splits = _fixed_split(text, chunk_size, overlap)

        elif strategy == ChunkStrategy.RECURSIVE:
            splits = _recursive_split(text, chunk_size, overlap, _RECURSIVE_SEPARATORS)

        elif strategy == ChunkStrategy.SEMANTIC:
            sentences = [s.strip() for s in text.split(". ") if s.strip()]
            groups = _semantic_split(sentences, _model, semantic_threshold)  # type: ignore[arg-type]
            splits = [". ".join(g) for g in groups]

        else:
            splits = [text]

        for i, split in enumerate(splits):
            if not split.strip():
                continue
            chunks.append(Chunk(
                content=split,
                metadata={**el.metadata, "chunk_index": i, "strategy": strategy.value},
                element_type=el.type,
            ))

    return chunks
