from functools import lru_cache
from app.core.config import get_settings

settings = get_settings()


class EmbeddingService:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer( settings.embedding_model, device=settings.embedding_device,)

    @property
    def dimension(self) -> int:
        dim = self.model.get_sentence_embedding_dimension()
        if not dim:
            raise RuntimeError("Embedding model did not expose vector dimension")
        return int(dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()


@lru_cache

def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
