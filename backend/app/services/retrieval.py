from app.core.config import get_settings
from app.schemas.retrieval import RetrievalFilters, RetrievalResponse, RetrievedEvidence
from app.services.embedding import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.reranker import get_reranker
from app.services.operation_lock import stable_kb
settings=get_settings()

class LegalRetriever:
    """Legal retrieval with metadata policy, dense evidence gate and optional cross-encoder reranking.
    The threshold must be calibrated only on a held-out pilot/dev set and then frozen for final experiments.
    """
    def __init__(self): self.embedding=get_embedding_service(); self.store=get_vector_store()
    @stable_kb
    def retrieve(self,query:str,top_k:int|None=None,filters:RetrievalFilters|None=None,rerank:bool|None=None)->RetrievalResponse:
        filters=filters or RetrievalFilters(); final_k=top_k or settings.final_top_k
        vector=self.embedding.embed_query(query); points=self.store.search(vector,limit=settings.retrieval_candidate_k,filters=filters)
        candidates=[]
        for point in points:
            p=point.payload or {}; meta=p.get('metadata',{}); score=float(point.score)
            # Evidence must first pass calibrated semantic threshold. Authority/status is then used as a conservative ranking prior.
            authority_bonus=0.02 if meta.get('is_official_source') is True else 0.0
            stale_penalty=0.08 if meta.get('status') in {'repealed','historical'} and not filters.include_historical else 0.0
            candidates.append({'payload':p,'dense_score':score,'accepted':score>=settings.dense_min_score,'rerank_score':None,'authority_adjustment':authority_bonus-stale_penalty})
        accepted=[c for c in candidates if c['accepted']]
        reranker=(get_reranker(force=True) if rerank is True else get_reranker()) if accepted and rerank is not False else None
        if accepted and reranker:
            pool=accepted[:max(final_k, settings.rerank_top_n)]; scores=reranker.score(query,[c['payload'].get('text','') for c in pool])
            for c,s in zip(pool,scores): c['rerank_score']=float(s)
            # Rank only candidates actually scored by the same cross-encoder.
            accepted=sorted(pool,key=lambda c:(c['rerank_score'],c['authority_adjustment']),reverse=True)
        else: accepted.sort(key=lambda c:c['dense_score']+c['authority_adjustment'],reverse=True)
        accepted=accepted[:final_k]; results=[]
        for rank,item in enumerate(accepted,1):
            p=item['payload']; meta=p.get('metadata',{})
            results.append(RetrievedEvidence(rank=rank,chunk_id=p['chunk_id'],document_id=p['document_id'],title=meta.get('title'),document_type=meta.get('document_type'),jurisdiction=meta.get('jurisdiction'),status=meta.get('status'),source_name=meta.get('source_name'),source_url=meta.get('source_url'),is_official_source=meta.get('is_official_source'),part=p.get('part'),schedule=p.get('schedule'),section=p.get('section'),article=p.get('article'),paragraph_ids=p.get('paragraph_ids',[]),page_start=p.get('page_start',0),page_end=p.get('page_end',0),dense_score=item['dense_score'],rerank_score=item['rerank_score'],accepted_as_evidence=True,text=p.get('text','')))
        return RetrievalResponse(query=query,retrieval_status='evidence_found' if results else 'insufficient_evidence',threshold_used=settings.dense_min_score,candidate_count=len(candidates),accepted_count=len(results),results=results, candidates=[{'rank':i+1, **c['payload'], 'dense_score':c['dense_score'], 'rerank_score':c['rerank_score'], 'passed_threshold':c['accepted']} for i,c in enumerate(candidates)])
_retriever=None
def get_retriever():
    global _retriever
    if _retriever is None:_retriever=LegalRetriever()
    return _retriever
