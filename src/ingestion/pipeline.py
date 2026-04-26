from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from ..embeddings.image_embedder import ImageEmbedder
from ..embeddings.text_embedder import TextEmbedder
from .chunker import ChunkStrategy, chunk_elements
from .parser import ElementType, parse_pdf

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "papers")

TEXT_DIM = 1024   # BAAI/bge-m3
IMAGE_DIM = 768   # CLIP ViT-L/14


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "text": VectorParams(size=TEXT_DIM, distance=Distance.COSINE),
            "image": VectorParams(size=IMAGE_DIM, distance=Distance.COSINE),
        },
    )
    print(f"Created collection '{COLLECTION}'")


def ingest_directory(
    input_dir: str | Path,
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = 300,
    overlap: int = 50,
    batch_size: int = 64,
) -> None:
    input_dir = Path(input_dir)
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _ensure_collection(client)

    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()

    for pdf_path in pdfs:
        print(f"\nProcessing: {pdf_path.name}")

        elements = parse_pdf(pdf_path)
        chunks = chunk_elements(elements, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
        print(f"  {len(elements)} elements → {len(chunks)} chunks")

        points: list[PointStruct] = []
        point_id = abs(hash(pdf_path.stem)) % (10**9)

        for chunk in chunks:
            text_vec = text_embedder.embed(chunk.content)

            if chunk.element_type == ElementType.IMAGE and chunk.image_path:
                image_vec = image_embedder.embed_image(chunk.image_path)
            elif chunk.element_type == ElementType.IMAGE:
                image_vec = image_embedder.embed_text(chunk.content)
            else:
                image_vec = [0.0] * IMAGE_DIM

            points.append(PointStruct(
                id=point_id,
                vector={"text": text_vec, "image": image_vec},
                payload={
                    "content": chunk.content,
                    "element_type": chunk.element_type.value,
                    "image_path": chunk.image_path,
                    **chunk.metadata,
                },
            ))
            point_id += 1

            if len(points) >= batch_size:
                client.upsert(collection_name=COLLECTION, points=points)
                points = []

        if points:
            client.upsert(collection_name=COLLECTION, points=points)

        print(f"  Indexed {len(chunks)} chunks for {pdf_path.stem}")

    print(f"\nDone. Collection '{COLLECTION}' ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument("--input", required=True, help="Directory containing PDFs")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "semantic"],
        default="recursive",
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    ingest_directory(
        input_dir=args.input,
        strategy=ChunkStrategy(args.strategy),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
