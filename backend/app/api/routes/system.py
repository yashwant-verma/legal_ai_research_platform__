from fastapi import APIRouter
from app.core.config import get_settings
from app.db.mongo import db
router=APIRouter(prefix='/system',tags=['system']); settings=get_settings()
@router.get('/config')
def config():
    return {'model':settings.ollama_model,'llm_provider':settings.llm_provider,'ollama_think':settings.ollama_think,'llm_max_output_tokens':settings.llm_max_output_tokens,'embedding_model':settings.embedding_model,'reranker_model':settings.reranker_model if settings.enable_reranker else None,'dense_min_score':settings.dense_min_score,'final_top_k':settings.final_top_k,'chunk_version':settings.chunk_version,'prompt_version':settings.prompt_version}
@router.get('/stats')
def stats():
    return {'documents':db.documents.count_documents({}),'chunks':db.chunks.count_documents({}),'datasets':db.datasets.count_documents({}),'experiments':db.experiments.count_documents({}),'evaluations':db.evaluations.count_documents({})}
