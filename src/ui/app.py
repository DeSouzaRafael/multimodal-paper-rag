from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from ..embeddings.text_embedder import TextEmbedder
from ..generation.llm_client import LLMClient
from ..retrieval.hybrid_search import HybridSearcher, RetrievalMode
from ..retrieval.reranker import Reranker
from ..retrieval.vector_store import SearchResult, VectorStore

st.set_page_config(
    page_title="Paper RAG",
    page_icon="📄",
    layout="wide",
)

# ── Session-level singletons ──────────────────────────────────────────────────

@st.cache_resource
def load_components():
    store = VectorStore()
    embedder = TextEmbedder()
    searcher = HybridSearcher(store=store, embedder=embedder)
    reranker = Reranker()
    llm = LLMClient()
    return searcher, reranker, llm


searcher, reranker, llm = load_components()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Retrieval settings")

    mode_label = st.selectbox(
        "Mode",
        options=[m.value for m in RetrievalMode],
        index=2,  # hybrid default
    )
    retrieval_mode = RetrievalMode(mode_label)

    top_k = st.slider("Top-K chunks", min_value=3, max_value=20, value=8)

    st.divider()
    st.subheader("Filters (optional)")
    filter_section = st.text_input("Section contains", placeholder="e.g. results")
    filter_paper = st.text_input("Paper ID", placeholder="e.g. 2310.12345")

    filters: dict | None = {}
    if filter_section:
        filters["section"] = filter_section
    if filter_paper:
        filters["paper_id"] = filter_paper
    if not filters:
        filters = None

    st.divider()
    st.caption("multimodal-paper-rag · phase 4")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Paper RAG")
st.caption("Ask questions about indexed scientific papers.")

query = st.text_input("Question", placeholder="What method did the authors use for evaluation?")

if st.button("Search", type="primary", disabled=not query):
    with st.spinner("Retrieving…"):
        chunks = searcher.search(
            query=query,
            mode=retrieval_mode,
            top_k=top_k,
            filters=filters,
            reranker=reranker if retrieval_mode == RetrievalMode.HYBRID_RERANK else None,
        )

    if not chunks:
        st.warning("No results found. Try a different query or check that papers are indexed.")
        st.stop()

    with st.spinner("Generating answer…"):
        result = llm.generate(query=query, chunks=chunks)

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.markdown(result.answer)

    # ── Retrieved chunks ─────────────────────────────────────────────────────
    st.subheader(f"Sources ({len(chunks)} chunks retrieved)")

    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        label = (
            f"[{i}] {meta.get('paper_id', 'unknown')} · "
            f"p.{meta.get('page', '?')} · "
            f"§{meta.get('section', '?')} · "
            f"{meta.get('element_type', 'text')} · "
            f"score {chunk.score:.3f}"
        )
        with st.expander(label, expanded=(i in result.citations)):
            if chunk.image_path and Path(chunk.image_path).exists():
                st.image(chunk.image_path, use_column_width=True)
            st.markdown(chunk.content)
