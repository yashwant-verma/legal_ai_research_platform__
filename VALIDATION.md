# Validation record

This release was validated without connecting to the user's MongoDB, Ollama service, PDFs or GPU.

- Backend: 62 tests passed with isolated/mocked services.
- Frontend: 9 Vitest tests passed.
- Frontend production build: `npm run build` passed.
- Setup document: rendered to four PNG pages and visually inspected.

The checks cover retrieval score separation, source and chunk provenance, reindex behavior, reasoning-output filtering, truncated-output rejection, experiment fingerprints, branch failure retention, multilingual grouping, reviewer persistence, language heuristics, threshold calibration, ingestion job UI and manual evaluation UI.

The following still require a local run before research results are reported: MongoDB migration, PDF reindexing, Ollama generation, GPU placement, threshold calibration, human review of multilingual gold answers and the final 20-group experiment.
