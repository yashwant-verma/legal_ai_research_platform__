from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator

SystemMode = Literal['llm_only','rag','compare','factual_compare']

class ExperimentCreate(BaseModel):
    benchmark_split: Literal['development','test'] = 'test'
    name: str
    dataset_id: str
    mode: SystemMode = 'compare'
    top_k: int = Field(default=5, ge=1, le=20)
    notes: str | None = None

class ManualEvaluationIn(BaseModel):
    correctness: float | None = Field(default=None, ge=0, le=1)
    groundedness: float | None = Field(default=None, ge=0, le=1)
    citation_correctness: float | None = Field(default=None, ge=0, le=1)
    hallucination_rate: float | None = Field(default=None, ge=0, le=1)
    evaluator: str | None = None
    notes: str | None = None

    error_categories: list[Literal['retrieval','generation','citation','multilingual','incomplete','other']] = Field(default_factory=list)

    @field_validator('correctness')
    @classmethod
    def binary_correctness(cls, value):
        if value is not None and value not in (0,1):
            raise ValueError('Correctness uses binary accuracy: 1 fully correct, 0 otherwise; explain partial answers in notes.')
        return value

class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    status: str
    dataset_id: str
    mode: str
    total_questions: int
    completed_questions: int
    failed_questions: int
    config_snapshot: dict
    aggregate: dict | None = None
