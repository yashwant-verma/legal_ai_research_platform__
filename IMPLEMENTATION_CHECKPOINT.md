# Implementation checkpoint — factual legal QA v5

Updated 2026-09-13. Implementation source:
`/workspace/scratch/1ffa7733f30a/factual_v5/legal_ai_research_platform`.
Baseline commit `b1d130c`: identical source to the v4 release. Intermediate implementation
commit `134f1e3`; implementation/validation commit `8b1d005`. Consult `git log` for final packaging records.

## Agreed scope

Seven domains: constitutional, criminal, cyber, consumer, contracts, property and selected family
laws. Fourteen selected central texts. English evidence and EN/HI/Hinglish question groups.
Gemini drafts factual candidates; one human group review creates the reference labels. Same
local Ollama model answers in LLM-only, RAG without reranking and RAG with reranking conditions.
Primary metrics are deterministic factual-answer accuracy, source recovery, latency and failures.
No LLM judge for primary scores; no semantic legal-correctness claim about explanations.
No MERN integration. Preserve existing chat, datasets, ingestion and manual evaluation.

## Implemented

- Typed factual schema, normalization/exact scorer, declared-scope prompt allow-list.
- Seven-domain/fourteen-Act catalog; Act/version/domain metadata fields and KB form templates.
- Source and quote/hash preflight, Mongo/Qdrant payload checks and active-edition checks.
- Per-request reranking switch without changing existing chat settings.
- Three-condition runner, frozen manifest/questions, failed-request resume, full attempt history.
- Final/candidate/pre-context retrieval metrics and failure-stage diagnosis.
- Overall/language/domain metrics, paired question-group bootstrap and CSV/JSON export.
- Gemini structured generation, per-language checkpoints, quotas/resume and locked-file recovery.
- One offline review screen with editable triplets/canonical answers, progress save and direct dataset export.
- Factual experiment UI/charts, source-check action, metadata editing inside the document modal.
- Offline paper table and PNG/SVG exporter.
- Complete setup/workflow guide and validation record.

## Verification

Backend: 92 tests pass. Frontend: 14 component/review tests and 5 stream tests pass.
Production frontend build passes (existing large-bundle advisory). See docs/VALIDATION_FACTUAL.md.
Gemini/Ollama/Mongo production services were mocked for integration tests; Qdrant point checks
also ran against an actual in-memory collection. Synthetic test plots are not paper results. The 35-file upgrade installer passed preflight,
apply/backups, data-preservation, idempotence, conflict refusal and injected-failure rollback tests.

## Remaining external work

1. Apply upgrade to the user's actual v4 directory after installer --check; resolve any reported
   user-source conflicts without deleting databases.
2. User verifies source editions/metadata and runs one Constitution group with their Gemini key.
3. Review that group; import; run local Ollama factual comparison and inspect real traces.
4. Once sources for other domains are ready, generate/review final and development groups,
   tune only on development data, freeze final experiment and export actual results.
5. Revise the paper using measured data; do not retain placeholder percentages.

## Delivery/persistence

Final deliverables are in `/workspace/scratch/1ffa7733f30a/delivery/factual_v5/`:
`apply_factual_upgrade.py`, `FACTUAL_SETUP_AND_WORKFLOW.md`, `IMPLEMENTATION_CHECKPOINT.md`,
`VALIDATION_FACTUAL.md`, `factual_v5_changes.patch` and `SHA256SUMS.txt`. The installer checks baseline hashes and backs up changed
code; user databases, PDFs, credentials and exports are not included or changed.
Library skill was read, but Library write tools are not exposed. Files are saved locally and must
be downloaded for durable user-side retention. Do not claim confirmed remote/durable storage.

## How to resume

Read this checkpoint and git status/log in the source directory. Continue from the verified state;
do not extract v4 over it. If the workspace has disappeared, request the saved upgrade script and
user's current project rather than pretending the old files remain accessible. The installer contains
all changed source files and can reconstruct this update on an unmodified v4 baseline.
