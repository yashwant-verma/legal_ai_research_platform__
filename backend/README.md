# Legal AI Research Platform — Backend v1.0

Backend for a controlled multilingual Indian legal-QA experiment comparing **LLM-only** and **evidence-gated legal RAG**. It is deliberately designed to abstain when the KB does not contain sufficiently relevant evidence.

## What is implemented

- FastAPI REST API + Swagger
- MongoDB research/document registry
- Qdrant vector store
- SHA-256 duplicate detection
- PDF text extraction with page traceability
- Optional Tesseract OCR fallback (English/Hindi)
- Conservative legal metadata with sidecar JSON support
- statute/article/section-aware and judgment-paragraph-aware chunking
- multilingual embeddings + optional cross-encoder reranker
- legal metadata filters, stale-law exclusion by default, official-source filter
- calibrated dense evidence gate and explicit abstention
- LLM-only, RAG-only, and compare-both APIs
- benchmark/custom dataset create/import/validate APIs (CSV/JSON/JSONL/XLSX)
- gold document/chunk labels for correct Hit@1/3/5 calculation
- MRR and gold-source recall retained in addition to paper metrics
- batch experiment runner with configuration snapshots
- manual research evaluation storage: correctness, groundedness, citation correctness, hallucination rate
- automatic citation-reference syntax validation
- result aggregation by language/domain
- CSV export
- knowledge-base document inspection, metadata update, OCR/reindex

## Core safety/research rule

A top-k result is **not automatically evidence**. Candidate chunks must pass the configured evidence threshold. If no candidate passes, the RAG branch returns an `insufficient_evidence` abstention instead of asking the LLM to improvise from unrelated text.

`DENSE_MIN_SCORE` is a development default only. Calibrate it on a separate pilot/dev set, freeze it, and only then run the final benchmark.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Keep the included .env; use your existing MongoDB instance.
# Start Ollama locally and ensure OLLAMA_MODEL is installed.
# Optional MongoDB Docker startup: docker compose up -d mongo
uvicorn app.main:app --reload --port 8000
```

Swagger: `http://localhost:8000/docs`

Optional backend Docker deployment (includes OCR packages; configure OLLAMA_BASE_URL to reach the host from the container):

```bash
docker compose -f docker-compose.full.yml up --build
```

## Important endpoints

### Knowledge base
- `POST /api/v1/kb/ingest`
- `GET /api/v1/kb/stats`
- `GET /api/v1/kb/documents`
- `GET /api/v1/kb/documents/{document_id}`
- `PUT /api/v1/kb/documents/{document_id}/metadata`
- `POST /api/v1/kb/documents/{document_id}/reindex`

### Retrieval/debugging
- `POST /api/v1/retrieval/search`

### Chat
- `POST /api/v1/chat/llm`
- `POST /api/v1/chat/rag`
- `POST /api/v1/chat/compare`
- `POST /api/v1/chat/llm/stream` (NDJSON token stream)
- `POST /api/v1/chat/rag/stream` (same evidence gate, NDJSON token stream)

### Datasets
- `GET/POST /api/v1/datasets`
- `POST /api/v1/datasets/import`
- `GET /api/v1/datasets/{dataset_id}`
- `GET /api/v1/datasets/{dataset_id}/questions`
- `GET /api/v1/datasets/{dataset_id}/validate`

### Experiments
- `GET/POST /api/v1/experiments`
- `POST /api/v1/experiments/{id}/run`
- `GET /api/v1/experiments/{id}`
- `GET /api/v1/experiments/{id}/items`
- `GET /api/v1/experiments/{id}/results`
- `PUT /api/v1/experiments/{id}/items/{question_id}/evaluation/{llm_only|rag}`
- `GET /api/v1/experiments/{id}/export.csv`

## Recommended legal metadata sidecar

For `some_act.pdf`, create `some_act.metadata.json` before ingestion:

```json
{
  "title": "Verified title",
  "document_type": "statute",
  "jurisdiction": "India",
  "authority_level": "central",
  "issuing_authority": "Verified authority",
  "act_name": "Verified act name",
  "year": 2024,
  "source_name": "Official source",
  "source_url": "https://...",
  "is_official_source": true,
  "effective_from": "2024-07-01",
  "effective_to": null,
  "version": "2024-07-01",
  "status": "in_force",
  "language": "en",
  "metadata_review_required": false
}
```

Do not mark heuristic metadata as verified. Version/status/effective dates are essential for legal retrieval.

## Dataset format and Hit@K

For each answerable benchmark question keep at least one verified `gold_document_ids` or `gold_chunk_ids`. If these are missing, dataset validation explicitly reports that Hit@K cannot be scored reliably.

The official 60-question design should use 20 legal intent groups × English/Hindi/Hinglish, with the same gold legal authority per group. Run `python scripts/create_benchmark_template.py` to create an empty 60-row template. It intentionally contains no invented legal answers.

## Evaluation

Automatic retrieval metrics:
- document/chunk Hit@1, Hit@3, Hit@5
- document/chunk Recall@K
- MRR
- abstention rate

Answer-level legal metrics are stored as human/research annotations. The backend does **not** pretend that lexical similarity equals legal correctness. Use the manual evaluation endpoint for correctness, groundedness, citation correctness, and hallucination rate. An LLM judge can be added later as a separate, clearly labelled experiment, not as hidden ground truth.

## Scaling to ~8,000 PDFs

Start with a reviewed 50–100 document subset. Validate extraction, metadata, legal chunk boundaries, evidence thresholds, and gold-source retrieval. Then scale ingestion. Use `scripts/ingest_folder.py` for batch processing and retain its ingestion report.

For production-scale ingestion, the next infrastructure upgrade should be a real job queue (Celery/RQ/Arq) rather than FastAPI background tasks. The current background experiment runner is appropriate for a research prototype and local deployment, not multi-worker distributed execution.

## Research reproducibility

Every experiment snapshots model, prompt version, embedding model, reranker, threshold, top-k, chunk version and vector collection. Do not alter these after looking at final benchmark results; create a new experiment/version instead.
