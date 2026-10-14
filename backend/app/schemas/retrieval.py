from typing import Literal
from pydantic import BaseModel, Field
class RetrievalFilters(BaseModel):
    document_type:str|None=None; jurisdiction:str|None=None; status:str|None=None; court:str|None=None; year:int|None=None
    official_only:bool=False
    include_historical:bool=False
class RetrievalRequest(BaseModel):
    query:str=Field(min_length=2); top_k:int=Field(default=5,ge=1,le=20); filters:RetrievalFilters|None=None
class RetrievedEvidence(BaseModel):
    rank:int; chunk_id:str; document_id:str; title:str|None=None; document_type:str|None=None; jurisdiction:str|None=None; status:str|None=None
    source_name:str|None=None; source_url:str|None=None; is_official_source:bool|None=None
    part:str|None=None; schedule:str|None=None
    section:str|None=None; article:str|None=None; paragraph_ids:list[str]=Field(default_factory=list)
    page_start:int; page_end:int; dense_score:float; rerank_score:float|None=None; accepted_as_evidence:bool; text:str
class RetrievalResponse(BaseModel):
    query:str; retrieval_status:Literal['evidence_found','insufficient_evidence']; threshold_used:float
    candidates:list[dict]=Field(default_factory=list)
    candidate_count:int=0; accepted_count:int=0; results:list[RetrievedEvidence]
