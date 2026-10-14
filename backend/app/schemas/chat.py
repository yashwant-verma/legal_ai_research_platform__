from pydantic import BaseModel, Field
from app.schemas.retrieval import RetrievalFilters, RetrievedEvidence


class ChatRequest(BaseModel):
    query: str = Field(min_length=2)
    language: str = "auto"
    filters: RetrievalFilters | None = None


class ChatAnswer(BaseModel):
    system_type: str
    answer: str
    abstained: bool = False
    prompt_version: str
    model: str
    retrieval_trace: dict | None = None
    # Non-fatal retrieval diagnostics.  These are shown as informational
    # notices in the UI and must not be treated as generation failures.
    notices: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence: list[RetrievedEvidence] = Field(default_factory=list)


class CompareResponse(BaseModel):
    query: str
    llm_only: ChatAnswer
    rag: ChatAnswer
