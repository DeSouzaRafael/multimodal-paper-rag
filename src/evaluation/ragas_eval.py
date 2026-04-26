from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from ..embeddings.text_embedder import TextEmbedder
from ..generation.llm_client import LLMClient
from ..retrieval.hybrid_search import HybridSearcher, RetrievalMode
from ..retrieval.reranker import Reranker
from ..retrieval.vector_store import VectorStore
from .test_dataset import QAPair, load_dataset

load_dotenv()

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _run_rag(
    pairs: list[QAPair],
    searcher: HybridSearcher,
    llm: LLMClient,
    mode: RetrievalMode,
    reranker: Reranker | None,
    top_k: int = 8,
) -> tuple[list[str], list[str], list[list[str]]]:
    questions, answers, contexts = [], [], []

    for pair in pairs:
        chunks = searcher.search(
            query=pair.question,
            mode=mode,
            top_k=top_k,
            reranker=reranker if mode == RetrievalMode.HYBRID_RERANK else None,
        )
        result = llm.generate(query=pair.question, chunks=chunks)
        questions.append(pair.question)
        answers.append(result.answer)
        contexts.append([c.content for c in chunks])

    return questions, answers, contexts


def evaluate_strategy(
    pairs: list[QAPair],
    mode: RetrievalMode,
    store: VectorStore | None = None,
    top_k: int = 8,
) -> dict:
    store = store or VectorStore()
    embedder = TextEmbedder()
    searcher = HybridSearcher(store=store, embedder=embedder)
    reranker = Reranker() if mode == RetrievalMode.HYBRID_RERANK else None
    llm = LLMClient()

    ground_truths = [p.answer for p in pairs]
    questions, answers, contexts = _run_rag(pairs, searcher, llm, mode, reranker, top_k)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = evaluate(dataset, metrics=METRICS)
    return scores.to_pandas().mean().to_dict()


def run_full_comparison(
    dataset_path: str | Path,
    output_path: str | Path = "data/processed/eval_results.json",
    modes: list[RetrievalMode] | None = None,
) -> pd.DataFrame:
    pairs = load_dataset(dataset_path)
    modes = modes or list(RetrievalMode)
    store = VectorStore()

    results = []
    for mode in modes:
        print(f"Evaluating mode: {mode.value} …")
        try:
            scores = evaluate_strategy(pairs, mode=mode, store=store)
            results.append({"mode": mode.value, **scores})
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({"mode": mode.value, "error": str(e)})

    df = pd.DataFrame(results)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)
    print(f"\nResults saved → {output_path}")
    print(df.to_string(index=False))

    return df
