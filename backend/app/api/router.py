from fastapi import APIRouter
from app.api.routes import health,kb,retrieval,chat,datasets,experiments,system
api_router=APIRouter()
for r in (health.router,kb.router,retrieval.router,chat.router,datasets.router,experiments.router,system.router): api_router.include_router(r)
