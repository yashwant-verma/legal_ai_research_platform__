import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatAnswer, CompareResponse
from app.services.chat import run_llm_only, run_rag, stream_llm_only, stream_rag
from app.services.llm.base import LLMError

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger("uvicorn.error.chat")


@router.post("/llm", response_model=ChatAnswer)
def llm_only(body: ChatRequest) -> ChatAnswer:
    return run_llm_only(body.query)


@router.post("/rag", response_model=ChatAnswer)
def rag(body: ChatRequest) -> ChatAnswer:
    return run_rag(body.query, body.filters)


@router.post("/compare", response_model=CompareResponse)
def compare(body: ChatRequest) -> CompareResponse:
    # Same user query, same configured model. Retrieval is the main branch difference.
    return CompareResponse(
        query=body.query,
        llm_only=run_llm_only(body.query),
        rag=run_rag(body.query, body.filters),
    )


# Existing JSON endpoints remain available to scripts and experiments.
# The chat UI uses NDJSON to display answer tokens as they arrive.


def _stream_response(events):
    def encode():
        try:
            for event in events:
                yield json.dumps(event, ensure_ascii=False) + '\n'
        except LLMError as exc:
            yield json.dumps({'type': 'error', 'detail': str(exc)}) + '\n'
        except Exception:
            logger.exception('Chat stream failed')
            yield json.dumps({'type': 'error', 'detail': 'The request failed. Check the backend terminal and that MongoDB and retrieval services are available.'}) + '\n'
        finally:
            events.close()
    return StreamingResponse(encode(), media_type='application/x-ndjson', headers={
        'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
    })


@router.post('/llm/stream')
def llm_only_stream(body: ChatRequest):
    return _stream_response(stream_llm_only(body.query))


@router.post('/rag/stream')
def rag_stream(body: ChatRequest):
    return _stream_response(stream_rag(body.query, body.filters))
