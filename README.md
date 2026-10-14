# Factual benchmark update (v5)

For the seven-domain factual workflow, use [Setup and workflow](docs/FACTUAL_SETUP_AND_WORKFLOW.md). The older guide below applies to existing chat/manual workflows. See [Resume checkpoint](IMPLEMENTATION_CHECKPOINT.md).

# Legal AI Research Platform

Standalone React + FastAPI + MongoDB + embedded Qdrant research platform using local Ollama. MERN integration is deferred. Existing chat, comparison, knowledge-base, dataset and experiment pages are retained.

## Upgrade an existing installation

Back up your project, backend/data, backend/.env and MongoDB database before replacing code. Stop the backend. Merge this archive into the existing project. This source release intentionally excludes user databases, PDFs, generated datasets, secrets, virtual environments and node_modules: keep your current copies. Do not delete backend/data or restore an older Qdrant snapshot over it.

Keep your MongoDB URI/database and vector collection settings. Compare backend/.env with backend/.env.example and set CHUNK_VERSION=legal_chunk_v2 and PROMPT_VERSION=legal_qa_v2 explicitly. Add the new context/device settings. CPU embeddings/reranking leave scarce VRAM for Ollama; this is a stability default, not a speed guarantee.

Run exactly one backend worker. Indexing, retrieval and experiment execution share a process lock; embedded Qdrant also requires a single owner. Do not run ingest_folder.py while the API is using embedded Qdrant. Synthetic dataset scripts use MongoDB read-only and may run with the API present, but avoid concurrent GPU work.

## Windows setup

From backend, with Python 3.11 or 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# New installations only; existing installations must preserve and edit their .env.
Copy-Item .env.example .env
ollama list
# Only if the model is missing:
ollama pull qwen3:4b
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

MongoDB and Ollama must be running. First embedding/reranker use downloads model weights and can take time. Separate terminal from frontend:

```powershell
npm ci
npm run dev
```

Open http://localhost:5173 and http://localhost:8000/docs. The health endpoint checks API liveness only.

## Required migration

In Knowledge base, inspect document metadata. For the Constitution use document_type=constitution and correct title, source/version/jurisdiction fields from the actual PDF. Do not guess dates or current legal status. Save metadata, then Reindex document. Saving metadata marks the document needs_reindex and excludes stale vectors until indexing completes. Watch the job state. OCR-required is not indexed.

Reindex changes chunk IDs/structure. Regenerate and review benchmark labels. Older experiment records remain readable; experiments without a v2 manifest cannot be resumed. Create new experiments after migration. Startup recovers interrupted runs as partial and interrupted ingestion as failed. Rerun only after checking configuration and source state.

## Multilingual benchmark

Recommended: generate and approve English base questions before translating them. All commands below run from backend.

```powershell
python scripts/synthetic_dataset.py generate --count 3 --languages en --model qwen3:4b --domain constitutional_law --output data/exports/base_01
```

Open base_01/review.html, review source support and answer completeness, approve valid bases, download decisions.json into base_01. Then:

```powershell
python scripts/synthetic_dataset.py translate --run data/exports/base_01 --decisions data/exports/base_01/decisions.json --output data/exports/triplets_01
```

Review all language variants in triplets_01/review.html, and download new decisions there. Edits require a new machine check and a fresh human review:

```powershell
python scripts/synthetic_dataset.py recheck --run data/exports/triplets_01 --decisions data/exports/triplets_01/decisions.json --output data/exports/rechecked_01
```

If you used recheck, use rechecked_01 and its NEW decisions for export. Otherwise:

```powershell
python scripts/synthetic_dataset.py export --run data/exports/triplets_01 --decisions data/exports/triplets_01/decisions.json --output data/exports/approved_01
```

Import approved_dataset.json in Datasets. Never add generated questions/answers to the knowledge base. Each group must have the same intent, answer facts, gold chunks and answerability in English, Devanagari Hindi and Roman Hinglish. Review heuristics can miss errors or reject valid phrasing; they are not language certification. Supporting quotes remain in the original source language.

A one-stage generate --languages en,hi,hinglish command is still supported. --count counts groups, not rows. 20 fully approved groups give 60 rows and 120 outputs when both systems run. Partial groups cannot export. Generate extra candidates if review rejects groups. Keep pilot and final questions/provisions separate. Translate/recheck save completed attempts; unlike generate --resume, these transitions do not support resume and require a fresh output folder after interruption.

## Threshold calibration

Use reviewed pilot questions only. Collect dense candidates, manually set each relevant field true/false after reading the full source, then score:

```powershell
python scripts/calibrate_retrieval.py collect --input data/exports/approved_01/approved_dataset.json --output data/exports/dev_labels.json
python scripts/calibrate_retrieval.py score --input data/exports/dev_labels.json --output data/exports/dev_threshold.json
```

The sweep suggests a candidate-level F1 threshold. Inspect query-level false acceptance/abstention as well, then set DENSE_MIN_SCORE in .env and restart before creating final experiments. The tool does not auto-change settings or certify calibrated legal relevance. Do not infer negative relevance merely from a chunk not appearing in the current gold set.

## Experiments and review

Import the reviewed dataset, Create experiment > Compare both > Top-K 5. Config/model digest/source hashes/code/dependencies are checked at start and finish; changes block mixing. Failed branches are retained separately; Resume failed items retries missing branches only. Execution order is alternated deterministically by group. Runtime timings include loading and retrieval and are not a controlled pure inference speed benchmark.

Inspect the run, select a question and system in Evaluate answers, enter evaluator name, scores and notes. Correctness is binary 0/1. Groundedness is the fraction of substantive claims supported by supplied evidence, generally N/A for LLM-only. Citation correctness is the fraction of checked claim-citation links supported by the passage. Hallucination is unsupported or contradicted substantive claims divided by substantive claims against reviewed authoritative references. Record claim counts and scope in notes. Leave non-applicable metrics blank. Multiple evaluators are retained and averaged per question before aggregation. These are human rubric scores, not automatic truthfulness guarantees.

Language tables report final-evidence Hit@1/3/5. Unique-document ranking removes repeated documents; chunk ranking retains chunk positions. Full JSON also includes dense candidates before gating and candidate metrics. Retrieval failures and missing gold are separate from misses; report denominators and errors. Paired correctness compares matched groups and includes an exploratory group bootstrap interval. RQ3 buckets show association between chunk Hit@1 and reviewed correctness, not causation.

Download Full JSON for complete manifest, answers, evidence, evaluations and errors; CSV is a flattened convenience export. Keep the reviewed dataset and review_audit.json with the experiment JSON. Full JSON is authoritative for all reviewer-level records.

## Tests and limits

Backend tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Runtime install already includes pytest. requirements-dev adds mongomock for isolated storage tests. Frontend production build uses npm run build. See VALIDATION.md for checks performed for this release.

No live Windows GPU, MongoDB, Ollama generation or legal accuracy experiment was run during the source revision. Thresholds, corpus authority and final legal answers still require local validation. Model names alone do not freeze Hugging Face caches: retain your downloaded embedding/reranker artifacts and environment for the final study. Context budgeting is based on characters, not an exact model tokenizer; inspect long prompts and increase context or use smaller evidence budgets if your Ollama build truncates inputs. The code detects output truncation and refuses to count it as a completed answer.

The bundled corpus from your earlier ZIP must be reindexed locally; this release does not fabricate a 60-row benchmark or research results. For the paper, replace completed MERN integration claims with the standalone architecture and future integration. Replace result placeholders only after actual evaluation.

Ollama API references: https://docs.ollama.com/api/chat and https://docs.ollama.com/capabilities/structured-outputs
