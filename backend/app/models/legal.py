from datetime import date
from typing import Literal
from pydantic import BaseModel, Field


DocumentType = Literal[
    "constitution",
    "statute",
    "rule",
    "regulation",
    "judgment",
    "notification",
    "circular",
    "order",
    "other",
]


class LegalMetadata(BaseModel):
    domain: str | None = None
    act_id: str | None = None
    snapshot_date: date | None = None
    downloaded_at: date | None = None
    amendment_coverage: str | None = None
    commencement_notes: str | None = None
    title: str | None = None
    document_type: DocumentType = "other"

    jurisdiction: str | None = None
    authority_level: str | None = None
    issuing_authority: str | None = None

    court: str | None = None
    bench: str | None = None
    judges: list[str] = Field(default_factory=list)
    case_title: str | None = None
    case_number: str | None = None
    neutral_citation: str | None = None
    decision_date: date | None = None

    act_name: str | None = None
    act_number: str | None = None
    year: int | None = None

    source_name: str | None = None
    source_url: str | None = None
    is_official_source: bool | None = None

    publication_date: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    version: str | None = None
    status: Literal["in_force", "amended", "repealed", "historical", "unknown"] = "unknown"

    language: str = "en"
    metadata_source: Literal["sidecar", "heuristic", "mixed"] = "heuristic"
    metadata_review_required: bool = True
    metadata_notes: list[str] = Field(default_factory=list)


class PageText(BaseModel):
    page_number: int
    text: str


class LegalChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_version: str
    text: str

    page_start: int
    page_end: int
    section: str | None = None
    subsection: str | None = None
    part: str | None = None
    schedule: str | None = None
    article: str | None = None
    paragraph_ids: list[str] = Field(default_factory=list)
    heading: str | None = None

    token_estimate: int
    metadata: LegalMetadata
