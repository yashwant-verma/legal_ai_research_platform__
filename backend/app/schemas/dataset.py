from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from app.schemas.factual import FactualLabel

Language = Literal['en','hi','hinglish','other']

class GoldAuthority(BaseModel):
    document_id: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    title: str | None = None
    act: str | None = None
    section: str | None = None
    article: str | None = None
    court: str | None = None
    citation: str | None = None
    version: str | None = None
    source_url: str | None = None

class DatasetQuestionIn(BaseModel):
    factual: FactualLabel | None = None
    question_id: str
    group_id: str | None = None
    language: Language = 'other'
    domain: str
    difficulty: Literal['easy','medium','hard'] | None = None
    question: str = Field(min_length=3)
    reference_answer: str | None = None
    gold_document_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_chunk_hashes: dict[str,str] = Field(default_factory=dict)
    gold_authorities: list[GoldAuthority] = Field(default_factory=list)
    answerable_from_kb: bool = True
    requires_multiple_sources: bool = False
    notes: str | None = None

    @model_validator(mode='after')
    def normalize_gold(self):
        docs = set(self.gold_document_ids)
        chunks = set(self.gold_chunk_ids)
        for a in self.gold_authorities:
            if a.document_id:
                docs.add(a.document_id)
            chunks.update(a.chunk_ids)
        self.gold_document_ids = sorted(docs)
        self.gold_chunk_ids = sorted(chunks)
        return self

class DatasetCreate(BaseModel):
    name: str
    version: str = 'v1.0'
    description: str | None = None
    benchmark: bool = False
    questions: list[DatasetQuestionIn] = Field(default_factory=list)

class DatasetSummary(BaseModel):
    dataset_id: str
    name: str
    version: str
    description: str | None = None
    benchmark: bool
    question_count: int
    language_counts: dict[str,int]
    domain_counts: dict[str,int]
    validation: dict

class DatasetValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    capabilities: dict[str,bool] = Field(default_factory=dict)
