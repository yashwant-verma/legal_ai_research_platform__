# Factual Legal QA — setup and workflow (v5)

This guide supersedes the older workflow for **factual benchmark experiments**. Existing chat,
legacy datasets and manual evaluation remain available. MERN integration is out of scope.

## 1. What this version does

- Seven legal domains and a catalog of fourteen selected central texts.
- One English evidence corpus; English, Hindi and Hinglish versions of the same questions.
- Gemini drafts candidates and translations from the actual indexed Mongo chunks.
- One group review page downloads directly importable reviewed JSON; there is no second
  question-level approval/export CLI stage.
- Ollama answers in three conditions: LLM-only, RAG without reranking and RAG with reranking.
- Deterministic factual-answer accuracy; no LLM judge for primary answer scores.
- Source-chunk recovery Hit@1/3/5, MRR, timing, errors, per-language/domain summaries,
  paired group bootstrap intervals, JSON/CSV and optional publication PNG/SVG exports.

These metrics do not establish explanation quality, legal truthfulness or suitability for legal advice.

## 2. Apply to your current v4 installation

Use the supplied `apply_factual_upgrade.py` against your existing project **root**, the folder
containing `backend` and `frontend`. No new full-project ZIP is required.

```powershell
python apply_factual_upgrade.py --root "D:\legal_ai_research_platform_updated_v4\legal_ai_research_platform" --check
python apply_factual_upgrade.py --root "D:\legal_ai_research_platform_updated_v4\legal_ai_research_platform"
```

Stop the backend/frontend before applying. The installer verifies every existing file against the
v4 baseline, refuses conflicting local modifications, saves backups, and never touches your
MongoDB, Qdrant data, uploaded PDFs, exports or local environment settings. If it reports conflicts,
keep the report and merge those files deliberately; do not delete your databases to fix a code conflict.
Running the installer again is safe when files already match this update.

## 3. Start the existing services

From your existing backend directory:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

MongoDB must be running and Ollama must be installed/running with your chosen model downloaded.
Use **one backend worker**. Embedded Qdrant is owned by that process; don't run separate ingestion
scripts that open the same Qdrant folder while the backend is running.

Keep your existing model; for your 4 GB GPU setup, the already tested `qwen3:4b` is a reasonable
initial development configuration. This is not a speed or correctness guarantee. Recommended
starting settings in backend `.env` (keep the actual model you have installed):

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_THINK=false
LLM_TEMPERATURE=0
OLLAMA_NUM_CTX=8192
RAG_CONTEXT_MAX_CHARS=16000
RETRIEVAL_CANDIDATE_K=20
RERANK_TOP_N=8
FINAL_TOP_K=5
```

Embedding/reranking weights may download on first use; subsequent runs can use cached local weights.

The factual experiment explicitly switches reranking per condition; it does not mutate your chat
reranking setting. Do not change thresholds/models/settings after freezing a final run.
If GPU loading fails, use the separately tested CPU configuration; measure and report that hardware.

In a second terminal, from frontend:

```powershell
npm ci
npm run dev
```

Open the Vite URL. Preserve the current backend `.env` and Qdrant path. A relative
`QDRANT_LOCAL_PATH=./data/qdrant` means the backend working directory matters. Starting from a
new root with an empty vector folder while reusing Mongo is a common source of missing hits.

## 4. Knowledge base scope and metadata

Catalog: `backend/config/legal_scope.json`.

| Domain | Stable Act IDs and texts |
|---|---|
| constitutional_law | COI — Constitution of India |
| criminal_law | BNS_2023, BNSS_2023, BSA_2023 — the three 2023 criminal statutes |
| cyber_law | IT_2000 — Information Technology Act, 2000 |
| consumer_law | CPA_2019 — Consumer Protection Act, 2019 |
| contract_law | ICA_1872 — Indian Contract Act, 1872; SRA_1963 — Specific Relief Act, 1963 |
| property_law | TPA_1882 — Transfer of Property Act, 1882; REG_1908 — Registration Act, 1908; RERA_2016 |
| family_law | HMA_1955 — Hindu Marriage Act; SMA_1954 — Special Marriage Act; PWDVA_2005 — Domestic Violence Act |

Use official text PDFs from India Code, the Legislative Department or issuing ministry.
This catalog lists selected texts; it does not certify that an arbitrary downloaded edition is
current. Verify amendments, commencement, omitted/invalidated provisions and applicable scope.
Rules-based/state-specific questions require their corresponding sources and are outside the
initial central-Act benchmark. Historical IPC/CrPC/Evidence Act versions must be separated from
current benchmark sources. Do not label a whole statute current merely because it was enacted.

In **Knowledge base**, select an Act template, enter the PDF edition/version and source URL,
verify legal status, and check the metadata review box only after reviewing. The JSON field
supports richer fields: `snapshot_date`, `downloaded_at`, `amendment_coverage`,
`commencement_notes`, authority, jurisdiction and source provenance.

For an existing uploaded PDF, open **Inspect**, edit metadata, **Save metadata**, then **Reindex
 document**. Wait for status `indexed`. Metadata changes require reindexing so Mongo and vector
payloads agree. Do not generate final labels before this is finished.

Keep one active English edition per Act in this benchmark. Mark older editions historical.
The prototype does not automatically research amendments or legal status for you.

## 5. Smoke-test the generator on the Constitution first

Open another terminal in backend and activate the environment. The generator reads Mongo only;
it does not open embedded Qdrant. Set your Gemini key locally without putting it in source code:

```powershell
$secureKey = Read-Host "Gemini API key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
python scripts/factual_dataset.py sources
python scripts/factual_dataset.py generate `
  --model YOUR_AVAILABLE_GEMINI_MODEL_ID `
  --domain constitutional_law `
  --groups-per-domain 1 `
  --dev-per-domain 0 `
  --output data/exports/factual_smoke_01
```

Replace `YOUR_AVAILABLE_GEMINI_MODEL_ID` with the valid model ID available to your key. Google
receives selected legal excerpts and prompts. Your experimental answering model remains Ollama.
The provider returns structured JSON; HTTP errors, non-JSON responses, rate limits and malformed
candidate output are reported separately. No API keys are written into checkpoints or exports.

Open `data/exports/factual_smoke_01/review.html`. Inspect the source and all three questions;
correct the question/reference/canonical value if needed, review aliases for short-text answers,
then approve the group. Enter your reviewer name and click **Download reviewed dataset**.
Import the resulting `reviewed_dataset.json` in the platform's **Datasets** page.

Editing a group clears its approval. Each export includes only approved complete groups.
Download review progress before closing; it is bound to the exact generated group collection.
Do not represent your own source check as independent legal-expert review.

## 6. Generate the balanced seven-domain collection

After all intended PDFs are indexed with reviewed metadata:

```powershell
python scripts/factual_dataset.py generate `
  --model YOUR_AVAILABLE_GEMINI_MODEL_ID `
  --groups-per-domain 10 `
  --dev-per-domain 2 `
  --max-attempts-per-domain 100 `
  --output data/exports/factual_seven_domains_01
```

This targets **70 test groups / 210 test rows**, plus **14 development groups / 42 development
rows**. Total drafting target: 84 groups / 252 rows. Groups are selected across source documents;
neighboring chunks from the same source provision are not split across development/test.
This is a small exploratory dataset, not a publication-mandated sample size.

If only some domains are ready, repeat `--domain` for those domains. A missing domain produces a
clear error rather than silently substituting Constitution questions. The generator checks language
format, verbatim supporting quotes, IDs, complete triplets and near duplicates. It does not certify
semantic translation equivalence; that's part of your one group review.

Files in the output folder:

- `factual_run.json`: authoritative resumable state, sources, partial translations and call metadata.
- `candidate_dataset.json`: draft rows; not ready for factual experiments.
- `review.html`: one offline group-review screen.
- `factual_run.json.recovery`: only if Windows blocked an atomic replace; resume selects the newest
  valid revision without truncating your older checkpoint.

A quota interruption or Ctrl+C: repeat the identical command with `--resume`. Completed languages
are reused. Successful groups are not regenerated. Source/config/code changes require a new run.
A group that exhausts validation retries is recorded as a failure and excluded; invalid rows are never
padded to reach the target. If the eligible source pool is insufficient, reduce the target or improve
sources, create a new run, and document the final coverage.

Regenerate just the review page, without any model calls:

```powershell
python scripts/factual_dataset.py review --run data/exports/factual_seven_domains_01
```

## 7. Canonical answer contract

Each dataset row retains the old fields and adds `factual` containing:

- `answer_type`: `provision`, `provision_set` or `short_text`.
- Declared `act_id`, `act_name` and `source_version` (public question scope).
- `canonical_answer`: Act ID, `provision_type` and values.
- `accepted_aliases`: reviewed translations/spellings for short text, specific to that language.
- `supporting_quotes`: exact quote for every gold chunk.
- `gold_label_kind`: `source_chunk` by default; use `reviewed_relevance` only after reviewing relevance labels.
- `split`, `review_status`, `reviewer`, `reviewed_at`, `generator_model`.

Provision answers must use explicit identifiers (`12`, `12(1)`, `21A`) rather than ranges.
The model receives only the question, language, answer type and declared Act/version scope.
It never receives reference answers, canonical values, aliases, quotes or gold IDs from the dataset.
For RAG, retrieved evidence is supplied by normal full-corpus search, not a gold-ID filter.

A correct number in an explanation does not count. Wrong Act/type, missing/extra provisions,
invalid JSON, abstention on these answerable questions and request failures are not correct.
Aliases are frozen before testing and must not be expanded after seeing test answers.

## 8. Run development, then freeze the test

1. Import the reviewed dataset. Use JSON for nested factual labels.
2. Click **Check factual sources and vectors**. It verifies IDs, text hashes, quotes, Act/version,
   metadata and Mongo/Qdrant payload agreement across the indexed corpus.
3. Open **Experiments**. Select **Factual comparison · 3 conditions**, split **Development / tuning**,
   top-K **5** (or more), and create the run.
4. Inspect failures: missing dense candidate, evidence-threshold rejection, ranking/top-K exclusion,
   context-budget exclusion, or gold in prompt with wrong/invalid answer.
5. Make justified configuration corrections using development data only.
6. Create a NEW experiment with split **Final test** and the frozen configuration.

The three conditions use the same local model, temperature, output budget and response schema.
Order uses six permutations selected deterministically per question group. Cache state is natural:
latency can include model/retriever loading. Report this, hardware and settings; this does not
pretend to provide an isolated GPU microbenchmark.

Experiments store a code/config/dataset/KB/model fingerprint and frozen questions. Resuming with
changed conditions blocks the run. Completed condition answers are reused; only request failures
are retried. Invalid but successfully returned output is a scored failure, not automatically retried
until correct. Full attempt history is preserved. Summary metrics use the latest attempt; report
retry counts and retain the initial failures in supplementary data.

After backend restart, click **Resume interrupted requests**. Single-worker startup marks
interrupted runs partial. A live run rejects duplicate starts. A blocked/completed run needs a new
experiment rather than overwriting results.

## 9. Metrics and denominators

| Metric | Definition |
|---|---|
| Factual accuracy | Exact/verified-alias matches / all selected test rows |
| Source recovery Hit@K | At least one labelled source chunk in final top-K evidence / eligible attempted rows |
| MRR | Mean reciprocal rank of first gold chunk, zero if absent within evaluated final depth |
| Failure rate | Request failures / attempted requests, per condition |
| Invalid-output rate | Malformed/inconsistent structured answers / successful responses |
| Median/p95 latency | End-to-end timing for attempted requests; latest attempts in summaries |
| Paired delta | Comparison accuracy minus baseline accuracy, grouped by underlying question |

All generated factual questions are answerable by construction and review. Retrieval failures
before a trace contribute a miss in the operational recovery metric and are counted separately.
Candidate and pre-context rankings are also retained. LLM-only retrieval metrics are null/N/A.
Incomplete/blocked experiments are explicitly provisional and rejected by the final paper exporter.

Confidence intervals resample whole question groups (2,000 bootstrap draws, seed 42). Three
translations are not three independent legal questions. Per-domain results from ten groups remain
exploratory. Factual accuracy does not measure semantic citation correctness or hallucinations.

## 10. Export tables and shareable figures

In the experiment detail download **Per-question CSV**, **Paper summary CSV** and **Full JSON**.
The full JSON includes frozen inputs, source rankings, responses, scores, errors, timings and
manifest. Keep it alongside your paper.

For offline PNG/SVG plots and paper tables:

```powershell
python -m pip install -r requirements-reporting.txt
python scripts/export_factual_paper.py `
  --bundle "C:\Users\YOUR_NAME\Downloads\EXP_YOUR_ID.json" `
  --output data/exports/paper_final_01
```

Outputs: `performance.csv`, `dataset_composition.csv`, `paired_comparisons.json`, `manifest.json`,
`language_accuracy.png/.svg`, `accuracy_latency.png/.svg` and metric interpretation notes.
No placeholder results are produced by this command. Blank CSV cells mean N/A, not zero.

For the paper: change scope to selected factual Indian legal QA, list only evaluated domains,
remove placeholder percentages and unsupported semantic-correctness claims, describe Gemini
assistance and reviewer identity/qualifications honestly, explain source-recovery labels and
limitations, and move MERN integration to future work. Publication is not guaranteed by software.

## 11. What was and was not validated here

Automated tests cover strict scores, Act mismatches, private-label exclusion, multilingual group
validation, stale source text/vector payloads, reranking overrides, request resume, frozen manifests,
Gemini response handling, partial-translation resume and Windows checkpoint recovery. Frontend
build/tests cover legacy modes and the new factual views. See `docs/VALIDATION_FACTUAL.md` for actual counts.

Your Windows GPU, local Ollama, Gemini account and real Mongo/Qdrant collection must be checked
with the one-group smoke run. No real legal benchmark scores are included in this update.

## 12. Resume implementation after a chat interruption

Keep `IMPLEMENTATION_CHECKPOINT.md` and the downloadable upgrade source. Say “resume” in this
conversation. If the working files still exist, work continues from the saved Git checkpoint.
A checkpoint cannot guarantee recovery after workspace deletion; keep downloaded copies locally.
The implementation checkpoint explicitly distinguishes completed work, verification and remaining
external smoke tests. Do not rebuild/delete your knowledge base solely because the chat paused.

## Developer regression commands

From backend:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-reporting.txt
python -m pytest -q
```

From frontend:

```powershell
npm ci
npm test
npm run build
```

Tests use local fixtures and mocked external services; fixture outputs are not paper results.
