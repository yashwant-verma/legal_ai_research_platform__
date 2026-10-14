from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import uuid
from qdrant_client import QdrantClient, models
from app.core.config import get_settings
from app.models.legal import LegalChunk
from app.schemas.retrieval import RetrievalFilters
from app.db.mongo import db
from app.services.embedding import get_embedding_service
settings = get_settings()

@lru_cache
def get_qdrant_client() -> QdrantClient:
    """
    Local embedded Qdrant client.
    No Docker or separate Qdrant server required.
    """
    path = Path(settings.qdrant_local_path)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))

class VectorStore:

    def __init__(self) -> None:
        self.client = get_qdrant_client()
        self.embedding = get_embedding_service()
        self.collection = settings.qdrant_collection
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(collection_name=self.collection):
            return
        self.client.create_collection(collection_name=self.collection, vectors_config=models.VectorParams(size=self.embedding.dimension, distance=models.Distance.COSINE))

    def upsert_chunks(self, chunks: list[LegalChunk]) -> None:
        if not chunks:
            return
        batch_size = 32
        total = len(chunks)
        print(f'[VECTOR] Total chunks to index: {total}')
        for start in range(0, total, batch_size):
            batch = chunks[start:start + batch_size]
            batch_end = min(start + len(batch), total)
            print(f'[VECTOR] Embedding chunks {start + 1}-{batch_end} of {total}...')
            texts = [chunk.text for chunk in batch]
            vectors = self.embedding.embed_documents(texts)
            points: list[models.PointStruct] = []
            for chunk, vector in zip(batch, vectors):
                payload = chunk.model_dump(mode='json')
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
                points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            print(f'[VECTOR] Indexed: {batch_end}/{total}')
        print(f'[VECTOR] Vector indexing complete: {total}/{total}')

    def delete_document(self, document_id: str) -> None:
        self.client.delete(collection_name=self.collection, points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key='document_id', match=models.MatchValue(value=document_id))])), wait=True)

    def _filter(self, filters: RetrievalFilters | None) -> models.Filter | None:
        filters = filters or RetrievalFilters()
        indexed = [d['document_id'] for d in db.documents.find({'ingestion_status':'indexed'}, {'document_id':1})]
        must: list[models.FieldCondition] = [models.FieldCondition(key='document_id', match=models.MatchAny(any=indexed))]
        must_not: list[models.FieldCondition] = []
        mapping = {'document_type': 'metadata.document_type', 'jurisdiction': 'metadata.jurisdiction', 'status': 'metadata.status', 'court': 'metadata.court', 'year': 'metadata.year'}
        for field, key in mapping.items():
            value = getattr(filters, field, None)
            if value is not None:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        if getattr(filters, 'official_only', False):
            must.append(models.FieldCondition(key='metadata.is_official_source', match=models.MatchValue(value=True)))
        include_historical = getattr(filters, 'include_historical', False)
        explicit_status = getattr(filters, 'status', None)
        if not include_historical and explicit_status is None:
            for bad_status in ('repealed', 'historical'):
                must_not.append(models.FieldCondition(key='metadata.status', match=models.MatchValue(value=bad_status)))
        if not must and (not must_not):
            return None
        return models.Filter(must=must, must_not=must_not)

    def search(self, query_vector: list[float], limit: int, filters: RetrievalFilters | None=None):
        response = self.client.query_points(collection_name=self.collection, query=query_vector, query_filter=self._filter(filters), limit=limit, with_payload=True)
        return response.points

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection, exact=True)
        return result.count

    def verify_chunks(self, chunks: list[dict]) -> list[str]:
        """Read exact points without embedding/search; detect a different local Qdrant DB."""
        errors = []
        for start in range(0, len(chunks), 128):
            batch = chunks[start:start+128]
            points = self.client.retrieve(self.collection,
                ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, c['chunk_id'])) for c in batch],
                with_payload=True, with_vectors=False)
            found = {(p.payload or {}).get('chunk_id'): p.payload for p in points}
            for c in batch:
                p = found.get(c['chunk_id'])
                if p is None: errors.append(f"{c['chunk_id']}: absent from Qdrant; reindex in this backend.")
                elif any(p.get(k) != c.get(k) for k in ('document_id','text','metadata')):
                    errors.append(f"{c['chunk_id']}: Mongo/Qdrant payload mismatch; reindex document.")
        return errors

    def collection_exists(self) -> bool:
        return self.client.collection_exists(collection_name=self.collection)

    def delete_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(collection_name=self.collection)

@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
