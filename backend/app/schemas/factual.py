"""Public question scope and private, reviewed factual scoring labels."""
from __future__ import annotations
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

DOMAINS = ('constitutional_law', 'criminal_law', 'cyber_law', 'consumer_law', 'contract_law', 'property_law', 'family_law')
AnswerType = Literal['provision', 'provision_set', 'short_text']

class CanonicalAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    act_id: str = Field(min_length=2)
    provision_type: Literal['article', 'section', 'rule', 'none']
    values: list[str] = Field(min_length=1)

    @model_validator(mode='after')
    def nonempty(self):
        self.values = [v.strip() for v in self.values]
        if any(not v for v in self.values): raise ValueError('Canonical values cannot be blank.')
        if self.provision_type != 'none' and any(not re.fullmatch(r'\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))*', v) for v in self.values):
            raise ValueError('Use explicit identifiers such as 12 or 12(1), not ranges/prose.')
        return self

class SupportingQuote(BaseModel):
    chunk_id: str
    quote: str = Field(min_length=10)

class FactualLabel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer_type: AnswerType
    act_id: str = Field(min_length=2)
    act_name: str = Field(min_length=3)
    source_version: str = Field(min_length=1)
    canonical_answer: CanonicalAnswer
    accepted_aliases: list[str] = Field(default_factory=list)
    supporting_quotes: list[SupportingQuote] = Field(min_length=1)
    gold_label_kind: Literal['source_chunk', 'reviewed_relevance'] = 'source_chunk'
    split: Literal['development', 'test'] = 'test'
    review_status: Literal['draft', 'verified'] = 'draft'
    reviewer: str | None = None
    reviewed_at: str | None = None
    generator_model: str | None = None

    @model_validator(mode='after')
    def consistent(self):
        if len(set(self.canonical_answer.values)) != len(self.canonical_answer.values): raise ValueError('Canonical answers cannot contain duplicate values.')
        if self.canonical_answer.act_id != self.act_id: raise ValueError('Canonical act_id must match question scope.')
        if self.answer_type != 'provision_set' and len(self.canonical_answer.values) != 1:
            raise ValueError('Single-answer tasks require exactly one canonical value.')
        if (self.answer_type == 'short_text') != (self.canonical_answer.provision_type == 'none'):
            raise ValueError('short_text requires type none; provisions require article/section/rule.')
        if self.answer_type != 'short_text' and self.accepted_aliases: raise ValueError('Aliases are only for short_text.')
        if self.review_status == 'verified' and (not (self.reviewer or '').strip() or not self.reviewed_at):
            raise ValueError('Verified labels need reviewer and timestamp.')
        return self

class FactualResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: CanonicalAnswer | None
    abstained: bool
    explanation: str = ''
    citations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def consistent(self):
        if self.abstained != (self.answer is None):
            raise ValueError('Abstention requires answer=null; an answer requires abstained=false.')
        return self
