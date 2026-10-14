from fastapi import APIRouter
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse
from app.services.retrieval import get_retriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalResponse)
def search(body: RetrievalRequest) -> RetrievalResponse:
    return get_retriever().retrieve(body.query, body.top_k, body.filters)
