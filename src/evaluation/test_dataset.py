from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv

from ..retrieval.vector_store import VectorStore

load_dotenv()

QuestionType = Literal["factual", "aggregation", "table", "figure"]

_GEN_PROMPT = """\
You are building an evaluation dataset for a RAG system over scientific papers.

Given the following paper chunk, generate {n} questions that can be answered using ONLY this chunk.
Vary the question types: factual (exact answer in text), aggregation (requires combining info), \
table (about tabular data), figure (about a figure or image).

Return a JSON array. Each item: {{"question": "...", "answer": "...", "type": "factual|aggregation|table|figure"}}

Chunk:
{chunk}"""


@dataclass
class QAPair:
    question: str
    answer: str
    type: QuestionType
    chunk_id: str | int
    paper_id: str
    section: str
    page: int


def generate_from_chunks(
    store: VectorStore,
    output_path: str | Path,
    questions_per_chunk: int = 2,
    max_chunks: int = 30,
    model: str = "claude-sonnet-4-6",
) -> list[QAPair]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    all_chunks = store.scroll_all()

    # sample evenly — prioritise tables and images for variety
    tables = [c for c in all_chunks if c.metadata.get("element_type") == "table"]
    images = [c for c in all_chunks if c.metadata.get("element_type") == "image"]
    texts  = [c for c in all_chunks if c.metadata.get("element_type") == "text"]

    # take up to 1/4 from tables and images, rest from text
    n_special = min(max_chunks // 4, len(tables) + len(images))
    special = (tables + images)[:n_special]
    text_sample = texts[: max_chunks - len(special)]
    sample = special + text_sample

    pairs: list[QAPair] = []

    for chunk in sample:
        prompt = _GEN_PROMPT.format(n=questions_per_chunk, chunk=chunk.content[:1200])
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            # extract JSON array even if surrounded by prose
            start = raw.find("[")
            end = raw.rfind("]") + 1
            items = json.loads(raw[start:end])
            for item in items:
                pairs.append(QAPair(
                    question=item["question"],
                    answer=item["answer"],
                    type=item.get("type", "factual"),
                    chunk_id=chunk.id,
                    paper_id=chunk.metadata.get("paper_id", "unknown"),
                    section=chunk.metadata.get("section", "unknown"),
                    page=chunk.metadata.get("page", 0),
                ))
        except Exception as e:
            print(f"Skipped chunk {chunk.id}: {e}")
            continue

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(p) for p in pairs], f, indent=2)

    print(f"Generated {len(pairs)} QA pairs → {output_path}")
    return pairs


def load_dataset(path: str | Path) -> list[QAPair]:
    with open(path) as f:
        return [QAPair(**item) for item in json.load(f)]
