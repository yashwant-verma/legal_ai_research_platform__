from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core.config import get_settings
from app.db.mongo import ensure_indexes
settings=get_settings()
@asynccontextmanager
async def lifespan(_:FastAPI): ensure_indexes(); yield
app=FastAPI(title=settings.app_name,version='1.0.0',lifespan=lifespan,description='Backend for multilingual Indian legal QA research: legal KB, evidence-gated RAG, datasets, experiments and evaluation.')
app.add_middleware(CORSMiddleware,allow_origins=[x.strip() for x in settings.cors_origins.split(',') if x.strip()],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
app.include_router(api_router,prefix=settings.api_prefix)


from fastapi.responses import JSONResponse
from app.services.llm.base import LLMError


@app.exception_handler(LLMError)
async def llm_error_handler(request, exc: LLMError):
    return JSONResponse(status_code=exc.status_code, content={'detail': str(exc)})
