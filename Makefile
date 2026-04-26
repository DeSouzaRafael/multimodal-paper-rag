.PHONY: up down ingest ui test eval

up:
	docker compose up -d

down:
	docker compose down

ingest:
	uv run python -m src.ingestion.pipeline --input data/pdfs/

ingest-semantic:
	uv run python -m src.ingestion.pipeline --input data/pdfs/ --strategy semantic

ui:
	uv run streamlit run src/ui/app.py

test:
	uv run pytest tests/ -v

eval:
	uv run python -c "from src.evaluation.ragas_eval import run_full_comparison; run_full_comparison('data/processed/qa_dataset.json')"

dataset:
	uv run python -c "from src.evaluation.test_dataset import generate_from_chunks; from src.retrieval.vector_store import VectorStore; generate_from_chunks(VectorStore(), 'data/processed/qa_dataset.json')"
