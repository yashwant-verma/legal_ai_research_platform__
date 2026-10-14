# Synthetic dataset workflow version 2

See ../../README.md (project root: ../../README.md from backend/scripts is actually ../../README.md) for the complete setup and commands.

1. Generate English bases with --languages en.
2. Approve their questions and complete answers against the source in review.html.
3. Use translate to generate Hindi and Hinglish versions of approved bases.
4. Review all three variants together. Their legal meaning and shared gold set must match.
5. If editing a question, answer or claims, download decisions and run recheck into a new directory. Recheck produces a NEW unapproved review snapshot.
6. Download fresh decisions for that snapshot, then export reviewed complete groups.
7. Import approved_dataset.json into Datasets, never into the knowledge base.

For transition failures, inspect run.json. Translate and recheck do not support --resume; use a new output directory for a fresh attempt. Generate supports --resume with unchanged settings and sources.

Default language generation is now en,hi,hinglish. Heuristic language checks reject obvious English-only Hinglish and non-Devanagari Hindi, but human review remains required. Source text hashes block changed gold chunks at experiment creation. Existing single-language generation remains supported.
