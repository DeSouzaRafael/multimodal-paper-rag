# multimodal-paper-rag

![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20progress-orange)
![Qdrant](https://img.shields.io/badge/vector--store-Qdrant-red?logo=qdrant&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-Claude%20API-blueviolet)

RAG system over scientific papers (PDFs) that handles text, tables, and figures together — not just raw text extraction.

Most RAG tutorials treat PDFs as plain text dumps. That breaks badly on real papers: multi-page tables get split, figure captions lose their context, equations turn into garbage. This project deals with those problems and compares different strategies along the way.

---

## What this does

- Parses PDFs into typed chunks: text blocks, tables (markdown/JSON), and images with their captions
- Embeds text with `sentence-transformers` and images with CLIP
- Stores everything in Qdrant with multi-vector support
- Supports dense, sparse (BM25), hybrid, and hybrid+rerank retrieval modes
- Evaluates all of it with RAGAS against a hand-curated question set
- Exposes a Streamlit UI with source citations and a retrieval mode toggle

---

## Stack

| Layer | Tool |
|---|---|
| Parsing | `unstructured` / LlamaParse (compared) |
| Text embeddings | `sentence-transformers` (BAAI/bge-m3) |
| Image embeddings | `open_clip` (CLIP ViT-L/14) |
| Vector store | Qdrant (local, Docker) |
| LLM | Claude API |
| Reranking | `bge-reranker-v2-m3` |
| Evaluation | RAGAS |
| UI | Streamlit |
| Tracing | Langfuse |

---

## Project structure

```
.
├── data/
│   ├── pdfs/           # raw arXiv papers
│   └── processed/      # extracted chunks
├── src/
│   ├── ingestion/      # parser, chunker, pipeline
│   ├── embeddings/     # text and image embedders
│   ├── retrieval/      # vector store, hybrid search, reranker
│   ├── generation/     # LLM client with citation prompting
│   ├── evaluation/     # RAGAS setup and test dataset
│   └── ui/             # Streamlit app
├── notebooks/          # exploration, chunking comparison, eval results
└── tests/
```

---

## Phases

**1 — Setup and exploration**
Spin up Qdrant, grab 10–20 arXiv papers across domains, and document the parsing challenges (multi-page tables, long captions, equations). Output: exploratory notebook.

**2 — Ingestion pipeline**
Implement a parser that extracts text blocks, tables, and images separately. Three chunking strategies: fixed-size with overlap, recursive by separators, and semantic (embedding-based topic boundaries). Each chunk carries metadata: paper id, section, page, content type.

Design decision for images: using CLIP embedding + caption embedding as separate vectors in Qdrant (multi-vector). Cheaper alternative would be caption-only + vision model description, but that loses visual similarity search. Tradeoff is documented here once benchmarked.

**3 — Retrieval**
Dense search → sparse (BM25) → hybrid with Reciprocal Rank Fusion → hybrid + cross-encoder rerank. Metadata filters (e.g., results section only) supported throughout.

**4 — Generation and UI**
Structured prompt that forces citations ([1], [2] linked to chunks). Streamlit app with a sidebar showing retrieved chunks, inline images, and a live toggle between retrieval modes.

**5 — Evaluation**
30–50 question test set (factual, aggregation, table-specific, figure-specific) generated with LLM and reviewed manually. RAGAS metrics: faithfulness, answer relevancy, context precision, context recall. Comparison table across chunking strategies and retrieval modes.

**6 — Portfolio polish**
Architecture diagram, results table, known failure analysis, demo GIF. Deploy on Hugging Face Spaces or Railway.

---

## Running locally

```bash
# start Qdrant
docker compose up -d

# install deps
uv sync

# copy and fill env
cp .env.example .env

# ingest papers
python -m src.ingestion.pipeline --input data/pdfs/

# run the UI
streamlit run src/ui/app.py
```

---

## Known issues / limitations

- Multi-page tables: `unstructured` splits them; current fix is post-processing by table title
- Equations: filtered out or parsed with Nougat — raw extraction is unusable
- CLIP embeddings run locally, which is slow on CPU (use GPU or reduce batch size)
- Hallucination still happens with low-quality retrieval — the prompt enforces citations but doesn't prevent fabrication when context is weak

---

## Evaluation results

_To be filled after Phase 5._

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| dense | — | — | — | — |
| hybrid | — | — | — | — |
| hybrid+rerank | — | — | — | — |

---

## References

- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — chunk positioning matters
- [RAGAS docs](https://docs.ragas.io)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [ColBERT paper](https://arxiv.org/abs/2004.12832)
