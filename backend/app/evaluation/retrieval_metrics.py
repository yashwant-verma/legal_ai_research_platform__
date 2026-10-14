from __future__ import annotations
from dataclasses import dataclass

def _ids(retrieved:list[dict],key:str)->list[str]: return [str(x.get(key)) for x in retrieved if x.get(key)]
def hit_at_k(gold:set[str],ranked:list[str],k:int)->float|None:
    if not gold:return None
    return 1.0 if gold.intersection(ranked[:k]) else 0.0
def recall_at_k(gold:set[str],ranked:list[str],k:int)->float|None:
    if not gold:return None
    return len(gold.intersection(ranked[:k]))/len(gold)
def reciprocal_rank(gold:set[str],ranked:list[str])->float|None:
    if not gold:return None
    for i,x in enumerate(ranked,1):
        if x in gold:return 1.0/i
    return 0.0
@dataclass
class RetrievalMetricResult:
    document_hit_at_1:float|None; document_hit_at_3:float|None; document_hit_at_5:float|None
    chunk_hit_at_1:float|None; chunk_hit_at_3:float|None; chunk_hit_at_5:float|None
    gold_document_recall_at_5:float|None; gold_chunk_recall_at_5:float|None
    document_mrr:float|None; chunk_mrr:float|None

def calculate_hit_at_k(retrieved_document_ids:list[str],retrieved_chunk_ids:list[str],gold_document_ids:list[str],gold_chunk_ids:list[str])->RetrievalMetricResult:
    retrieved_document_ids=list(dict.fromkeys(retrieved_document_ids))
    gd=set(gold_document_ids);gc=set(gold_chunk_ids)
    return RetrievalMetricResult(hit_at_k(gd,retrieved_document_ids,1),hit_at_k(gd,retrieved_document_ids,3),hit_at_k(gd,retrieved_document_ids,5),hit_at_k(gc,retrieved_chunk_ids,1),hit_at_k(gc,retrieved_chunk_ids,3),hit_at_k(gc,retrieved_chunk_ids,5),recall_at_k(gd,retrieved_document_ids,5),recall_at_k(gc,retrieved_chunk_ids,5),reciprocal_rank(gd,retrieved_document_ids),reciprocal_rank(gc,retrieved_chunk_ids))

def score_retrieval(gold_document_ids:list[str],gold_chunk_ids:list[str],retrieved:list[dict],ks=(1,3,5))->dict:
    docs=list(dict.fromkeys(_ids(retrieved,'document_id')));chunks=_ids(retrieved,'chunk_id');gd=set(gold_document_ids);gc=set(gold_chunk_ids);out={}
    for k in ks:
        out[f'document_hit_at_{k}']=hit_at_k(gd,docs,k);out[f'chunk_hit_at_{k}']=hit_at_k(gc,chunks,k)
        out[f'document_recall_at_{k}']=recall_at_k(gd,docs,k);out[f'chunk_recall_at_{k}']=recall_at_k(gc,chunks,k)
    out['gold_document_recall_at_5']=recall_at_k(gd,docs,5);out['gold_chunk_recall_at_5']=recall_at_k(gc,chunks,5)
    out['document_mrr']=reciprocal_rank(gd,docs);out['chunk_mrr']=reciprocal_rank(gc,chunks)
    return out
