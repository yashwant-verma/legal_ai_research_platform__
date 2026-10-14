# Synthetic dataset generation — local Ollama, no API key

This is a separate command-line workflow. It does not change the app, `.env`, model configuration, MongoDB records, Qdrant index, existing datasets or experiments. MongoDB is read-only. The script uses the backend's current `.env` even when invoked from another directory. It imports no embedding/reranker models and does not open the embedded Qdrant directory.

The output is a **candidate benchmark requiring review**, not automatically verified legal ground truth. Do not import generated answers into your knowledge base.

## Copy into your current project

From this tools ZIP, copy these three files into your existing `backend/scripts` folder:

- `synthetic_dataset.py`
- `synthetic_review.html`
- `SYNTHETIC_DATASET_GUIDE.md`

Optional: copy `backend/tests/test_synthetic_dataset.py` for regression tests.

Keep your existing `backend/data` and `.env`. Do not overwrite your newer local database/data with the ZIP's old snapshot. Existing app dependencies (`httpx`, `pymongo`, `pydantic`, `pydantic-settings`) are sufficient; no new application package is needed.

## 1. Try a three-question pilot first

Activate your existing virtual environment. Keep MongoDB and Ollama running. The frontend and backend API do not need to be running for this script. Avoid concurrent chat/ingestion while generating on a small GPU.

From your project folder:

```powershell
cd backend
python scripts/synthetic_dataset.py generate --count 3 --languages hinglish --model qwen3:4b --output data/exports/synthetic_trial_01
```

Use a model already installed locally. Omit `--model` to use the model in your current `.env`. The flag applies only to this script. It does not switch the chat app's model. If needed, `ollama pull qwen3:4b` installs that model separately.

The script reads chunks from all documents marked `indexed`. To restrict it to one PDF, append its actual registry ID:

```powershell
python scripts/synthetic_dataset.py generate --document-id ACTUAL_DOC_ID --count 3 --languages hinglish --model qwen3:4b --output data/exports/constitution_trial_01
```

Replace `ACTUAL_DOC_ID`; do not type that placeholder literally. Repeat `--document-id` for several PDFs.

Console messages show the source chunk, drafting, verification and any rejection/error. Every completed attempt is checkpointed. There are two Ollama calls per successful language row (draft/translation and verification), so this is slower than one chat answer. No particular throughput or acceptance count is guaranteed on your hardware.

## 2. Review in your browser

Open the generated `review.html` by double-clicking it, or from PowerShell:

```powershell
Start-Process .\data\exports\synthetic_trial_01\review.html
```

Each candidate shows:

- Question and draft reference answer.
- All declared answer claims and their exact source excerpts.
- The full original chunk, not just a retrieved snippet.
- Actual document/chunk IDs, page/article/section, hashes, source version and metadata.
- The model verifier's assessment and whether it was the same model as the generator.
- Possible additional gold chunks containing every quoted supporting excerpt.

Enter your reviewer name. Select **Approve** only after checking accuracy, completeness, source/version scope, language and supporting evidence. Reject unsuitable questions; this version deliberately does not silently repair a rejected answer into an approved one.

For possible additional gold chunks, read their full text before selecting them. They are suggestions only. The anchor chunk is the default gold; alternatives are added only if you explicitly check them. The search is limited to exact-quote overlap within the same document. It does not discover every paraphrased or cross-document equivalent provision. A reviewer should still assess gold completeness before a final benchmark.

Machine-rejected candidates cannot be approved for export. Source metadata marked `unknown`/review-required is displayed; verify it instead of assuming the text is current law. The page renders source/model text safely as text, without executing HTML from PDFs.

Click **Download decisions.json** and save that file into the run directory. Your browser may put it in Downloads first; move it manually. Browser-local saves are a convenience, not the authoritative export. You can load a downloaded decisions file back into the same review snapshot.

## 3. Export approved questions

```powershell
python scripts/synthetic_dataset.py export --run data/exports/synthetic_trial_01 --decisions data/exports/synthetic_trial_01/decisions.json --output data/exports/approved_trial_01
```

MongoDB must still be running. The script checks that the approved sources still exist, are indexed, and match the reviewed text/metadata snapshot. A changed source blocks export; generate a new run instead of reusing stale labels.

Output:

- `approved_dataset.json`: preferred app import; includes structured gold authorities.
- `approved_dataset.csv`: UTF-8 with BOM; lists of IDs separated by semicolons, compatible with the current importer.
- `review_audit.json`: provenance, full source snapshots, checks and human review decisions.

Import **only** `approved_dataset.json` (or the CSV alternative) through **Datasets → Import**. Do not import `run.json`, `candidates.json`, `decisions.json` or the audit file. They are not dataset files. Then select the dataset in **Experiments → Compare both → Top-K 5 → Create and run**.

Exports require a new or empty output directory and at least one approved row. Draft rows cannot be exported just by setting a generic `approved=true` field; the matching review decisions and source snapshot are required.

## 4. Generate a larger candidate pool

After the three-question trial succeeds:

```powershell
python scripts/synthetic_dataset.py generate --count 30 --languages hinglish --model qwen3:4b --output data/exports/synthetic_pilot_01
```

`--count 30` targets 30 machine-passed source-question groups. It is not a promise of 30 human-approved questions. Sampling is seeded and round-robin across documents. Empty, oversized, obvious contents-list, low-letter-ratio, replacement-character-heavy and exact-duplicate passages are filtered. Remaining OCR defects and incomplete legal provisions still need review. Oversized passages are skipped, never silently sliced.

Choose 10 strong, verified questions for the initial pilot. Use a new output directory when changing settings or source versions.

For multilingual groups:

```powershell
python scripts/synthetic_dataset.py generate --count 20 --languages en,hi,hinglish --model qwen3:4b --output data/exports/multilingual_candidates_01
```

The first language is the base question; subsequent languages translate the same question and answer from the same source. They share a `group_id` and have different `question_id` values. Every language gets a separate verification pass. Translation groups can be incomplete when a candidate is rejected; export requires all requested language variants of each included group to be approved. Reject the entire group if one variant is unsuitable.

Twenty fully approved groups produce 60 rows. These remain 20 underlying questions, not 60 independent legal questions.

For `qwen3:4b`, the more reliable workflow is to generate the English base first and translate approved candidates in a second pass. A one-step `en,hi,hinglish` run is supported, but the model may answer a Hinglish prompt in plain English; the language validator correctly flags that row instead of silently accepting it. These flags mean the language draft failed review, not that the source chunk or MongoDB lookup failed.

## Resume and troubleshooting

For an interrupted run, repeat the SAME generation command and append `--resume`:

```powershell
python scripts/synthetic_dataset.py generate --count 3 --languages hinglish --model qwen3:4b --output data/exports/synthetic_trial_01 --resume
```

Completed attempts are retained. A transient Ollama failure stops the run rather than making the same failed API call for every chunk; failed runtime attempts are retried on resume and retained in failure history. An in-flight interrupted request may need to be generated again. Settings or source changes require a new run directory. Do not resume a run while another process is writing to it.

- Missing model / Ollama unavailable: check `ollama list`, start Ollama, then resume.
- MongoDB unavailable: use the same working MongoDB service and URI as your app.
- GPU failure: an optional `--num-gpu 0` runs the script's requests on CPU; use a NEW run directory because this changes the run configuration. CPU generation can be slow.
- Context errors: default `--num-ctx 8192` includes source, instructions and output. The request disables input truncation. Adjust `--max-source-chars` downward to skip long chunks or use a larger context if hardware allows. Any setting change needs a new run.
- Output token truncation: candidate rejected. `--tokens 1800` is the default; increasing it may help valid structured answers, at a speed/memory cost.
- Long requests: default `--timeout 300` is an HTTP timeout; Ctrl+C checkpoints completed work. `--max-attempts 150` bounds candidate attempts.
- Repeated poor candidates: inspect `run.json` errors and the original OCR text. Do not weaken checks just to reach the requested count.
- Windows `WinError 5` while replacing `candidates.json`, `run.json` or `review.html`: close that file in VS Code, a browser tab, Explorer Preview, OneDrive/sync software or antivirus quarantine, then rerun the same command with `--resume`. The writer retries transient locks and writes a `.recovery` copy if a persistent lock remains. If the directory is continuously watched, use a new output directory; do not delete the existing run.

## Research limitations

1. Questions and reference answers are synthetic drafts. Exact quote matches verify provenance, not logical entailment or complete legal correctness. A separate LLM verifier is helpful but is not a legal authority. Human review remains required.
2. The default verifier is the same local model to avoid downloading/loading another model on limited hardware. This is explicitly recorded and is not independent evaluation. Use `--verifier-model INSTALLED_MODEL_NAME` if appropriate; model-switching may increase latency.
3. This version generates **single-source, answerable questions**. It does not fabricate unanswerable questions, generate multi-hop cases, certify temporal/legal status, or compute answer-quality scores. Create a separately verified unanswerable set and a multi-source set for the final study.
4. Near-duplicate question detection is lexical, not a guarantee of different legal intent. Before train/dev/test splitting, group translated, overlapping and same-provision questions together. Prefer grouping by legal provision/document when appropriate, beyond the generated `group_id`.
5. Freeze source/chunk versions, settings and reviewed benchmark before final experiments. Model names and request settings are recorded; names alone do not guarantee identical model weights across future downloads. Retain your Ollama model version/digest externally for publication reproducibility.
6. Never put synthetic question/reference-answer files into the RAG knowledge base. Keep them in datasets/exports.
7. The export contains document IDs, chunk IDs, provenance and reference answers for automatic retrieval scoring. Correctness, groundedness and citation correctness still require the existing manual evaluation workflow.

## Validation of this implementation

Regression tests exercise actual existing dataset import parsing, source ID ownership, quote validation, reasoning-text rejection, stale/tampered decisions, explicit approval, alternative gold labels, language grouping, checkpoint/resume, source filtering and read-only MongoDB calls. The offline review page was exercised in jsdom for rendering, approval, rejected-row blocking, filtering, decisions download and safe source display. Ollama and MongoDB services were mocked; no benchmark was generated from your live local database here.

```powershell
# From backend, using your existing environment
python -m pytest tests/test_synthetic_dataset.py -q
python scripts/synthetic_dataset.py --help
```

The structured request follows [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
