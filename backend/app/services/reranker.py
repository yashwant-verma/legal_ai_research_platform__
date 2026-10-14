from functools import lru_cache
import numpy as np
from app.core.config import get_settings

settings = get_settings()


class Reranker:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(settings.reranker_model, device=settings.reranker_device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = np.asarray(self.model.predict([(query, text) for text in texts]), dtype=float).reshape(-1)
        # Keep raw score for ranking; do not pretend it is universally calibrated probability.
        return raw.tolist()


@lru_cache

def get_reranker(force: bool = False) -> Reranker | None:
    if not force and not settings.enable_reranker:
        return None
    return Reranker()
