COMBINED FIXES

This archive combines the three requested fixes:

1. RAG context notices
   - Context-budget exclusions are stored in retrieval_trace.context_excluded_chunks.
   - They are shown as informational notices, not red generation errors.
   - True citation warnings, no-evidence abstention, and LLM failures are unchanged.

2. Factual neutral baseline
   - factual_chat.py uses factual_json_schema_v3_neutral.
   - Numeric examples such as 12 and 12(1) were removed from the baseline prompt.
   - Provision output values are constrained to bare identifier tokens, so a
     label such as "Article" cannot be returned inside the value field.
   - factual_experiments.py evaluation, scoring, conditions, and retry logic are unchanged.

3. Frontend production build
   - The rebuilt frontend dist is included so the notice styling is available in a production build.

IMPLEMENTATION

Extract this archive into the project root: the folder that contains both
backend and frontend. Allow files to merge/replace at these exact paths:

backend/app/schemas/chat.py
backend/app/services/chat.py
backend/app/services/factual_chat.py
backend/app/services/factual_experiments.py
frontend/src/components/ChatPanel.jsx
frontend/dist/*

PowerShell example:

Expand-Archive .\combined_fixes_v1.zip -DestinationPath . -Force

Then restart the backend. If using the Vite development server, restart
`npm run dev`. If serving the production build, the included frontend/dist is
already rebuilt; no source-code changes outside the listed files are required.

IMPORTANT

- Create a NEW factual experiment after installing the neutral prompt. Do not
  resume an experiment created with factual_json_schema_v2.
- Keep OLLAMA_THINK=false and restart the backend after changing .env.
- The factual_chat.py file calls stream(..., response_format=...). Therefore
  the installed OllamaClient must be the structured-JSON version that accepts
  the response_format argument.
- RAG_CONTEXT_MAX_CHARS remains unchanged; retrieval and benchmark behavior are
  not silently altered.

VALIDATION

The patched Python files compile successfully and the frontend production build
completed successfully.
