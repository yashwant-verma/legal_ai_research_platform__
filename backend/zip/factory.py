# backend/app/services/llm/factory.py

from app.core.config import get_settings
from app.services.llm.ollama_client import get_ollama_client

settings = get_settings()


def get_llm_client():

    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return get_ollama_client()

    raise RuntimeError(
        f"Unsupported LLM provider: {provider}"
    )