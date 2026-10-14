from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = "Legal AI Research Platform"
    app_env: str = "development"
    api_prefix: str = "/api/v1"

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "legal_ai_research"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "legal_chunks_v1"
    qdrant_local_path: str = "./data/qdrant"
    embedding_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 16
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"
    ollama_num_ctx: int = 8192
    ollama_num_gpu: int | None = None
    rag_context_max_chars: int = 8000

    enable_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_candidate_k: int = 20
    dense_min_score: float = 0.48
    rerank_top_n: int = 8
    final_top_k: int = 5

    chunk_target_tokens: int = 850
    chunk_max_tokens: int = 1200
    chunk_overlap_tokens: int = 120
    chunk_version: str = "legal_chunk_v2"

    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_think: bool = False
    ollama_keep_alive: str = "10m"
    ollama_timeout_seconds: float = Field(default=180.0, gt=0)
    ollama_max_duration_seconds: float = Field(default=300.0, gt=0)

    llm_max_output_tokens: int = 2048
    llm_temperature: float = 0.0
    prompt_version: str = "legal_qa_v2"

    upload_dir: Path = Path("./data/uploads")
    enable_ocr: bool = False
    ocr_languages: str = "eng+hin"
    ocr_dpi: int = 220
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache

def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
