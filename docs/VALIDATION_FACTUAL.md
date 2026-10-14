# Factual update validation

Date: 2026-09-13. Baseline: v4, Git commit `b1d130c`.

## Verified locally

- Backend: 92 tests pass, including 30 new factual/generation/export cases.
- Frontend: 14 component/review-page tests and 5 stream tests pass.
- Frontend production build succeeds. Existing large-bundle warning remains (about 749 kB
  uncompressed JavaScript); no functional build errors.
- Python compilation and Git whitespace checks pass.
- Canonical scoring covers wrong Act/type, extra/missing values, duplicate sets, verified aliases,
  Devanagari digits, malformed JSON and correct numbers appearing only in explanations.
- Prompt allow-list excludes private benchmark answers, aliases, quotes and source IDs.
- Mongo source ownership/hash/quote checks and actual in-memory Qdrant point verification pass.
- Reranking override preserves global chat configuration; stages retain candidate/pre-context/final rankings.
- Three-condition mocked experiment resumes only failed requests, retains attempt history and blocks
  changed datasets. Failure/invalid-output denominators and group-based comparisons are exercised.
- Mocked Gemini structured response, HTML HTTP error handling, partial-translation resume and
  Windows locked-file recovery pass. No real Gemini key was used.
- Offline review page requires named reviewer and individual approval; editing clears approval.
- Paper exporter writes fixture CSV/PNG/SVG and refuses incomplete experiments. Fixture figures
  are test data and are not supplied as measured paper results.
- Source-bearing installer: all 35 packaged files verified against the built source. Read-only
  preflight, application/backups, user-data preservation, idempotent rerun, conflict refusal with
  no writes, and rollback after an injected write failure all pass against a clean v4 fixture.
  Filesystem tests ran on Linux; Windows locking recovery was separately simulated.

## External smoke test still required

No access to the user's Windows GPU, installed Ollama runtime/model, Gemini account or local
MongoDB/Qdrant collection was used. Run the one-group Constitution workflow in the setup guide
before generating a large dataset. Exact GPU timing and live Gemini model compatibility remain
unverified here. Legal editions, supporting references and translations require the agreed dataset review.

## Preserved functionality

Existing LLM-only/RAG/compare chat, PDF ingestion, metadata review/reindexing, legacy dataset
formats, manual evaluation and legacy experiment exports remain supported. No MERN integration,
PDF deletion, database reset, automatic approval of legal labels or fabricated experiment scores.
