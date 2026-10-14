# Legal AI Research Platform — Frontend

React + Vite frontend connected to the FastAPI backend.

## Pages
- Dashboard
- LLM-only chat
- Evidence-gated RAG chat
- Compare both branches
- Knowledge Base ingestion/registry
- Dataset import/validation
- Experiment creation, progress and results

## Run
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```
Default API: `http://localhost:8000/api/v1`

## Backend
Run the backend on port 8000. Its default CORS setting already allows `http://localhost:5173`.
