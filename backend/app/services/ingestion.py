from __future__ import annotations
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from app.core.config import get_settings
from app.db.mongo import db
from app.services.extraction import extract_pdf_pages, needs_ocr
from app.services.legal_metadata import resolve_metadata
from app.services.chunking import chunk_document
from app.services.vector_store import get_vector_store
from app.services.ocr import ocr_pdf_pages
from app.services.operation_lock import stable_kb
import json
settings = get_settings()

def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            d.update(block)
    return d.hexdigest()

def _index(document_id: str, document_hash: str, source_path: Path, stored_path: Path, pages, extraction_method: str) -> dict:
    try:
        sidecar_src = source_path.with_suffix('.metadata.json')
        sidecar_dst = stored_path.with_suffix('.metadata.json')
        if sidecar_src.exists() and sidecar_src.resolve() != sidecar_dst.resolve():
            shutil.copy2(sidecar_src, sidecar_dst)
        metadata = resolve_metadata(stored_path, pages)
        chunks = chunk_document(document_id, pages, metadata)
        if not chunks:
            raise RuntimeError('No retrievable chunks were produced from PDF')
        now = datetime.now(timezone.utc)
        record = {'document_id': document_id, 'document_hash': document_hash, 'filename': source_path.name, 'stored_path': str(stored_path), 'ingestion_status': 'indexing', 'extraction_method': extraction_method, 'created_at': now, 'updated_at': now, 'page_count': len(pages), 'chunk_count': len(chunks), 'chunk_version': settings.chunk_version, 'embedding_model': settings.embedding_model, 'metadata': metadata.model_dump(mode='json')}
        db.documents.update_one({'document_id': document_id}, {'$set': record}, upsert=True)
        db.chunks.delete_many({'document_id': document_id})
        chunk_documents = [chunk.model_dump(mode='json') for chunk in chunks]
        batch_size = 200
        inserted_count = 0
        for start in range(0, len(chunk_documents), batch_size):
            batch = chunk_documents[start:start + batch_size]
            if not batch:
                continue
            result = db.chunks.insert_many(batch, ordered=True)
            inserted_count += len(result.inserted_ids)
            print(f'[INGESTION] MongoDB chunks: {inserted_count}/{len(chunk_documents)}')
        saved_chunk_count = db.chunks.count_documents({'document_id': document_id})
        if saved_chunk_count != len(chunks):
            raise RuntimeError(f'MongoDB chunk insert incomplete. Expected {len(chunks)} chunks but found {saved_chunk_count}.')
        print('[INGESTION] All MongoDB chunks saved:', saved_chunk_count)
        vector_store = get_vector_store()
        vector_store.delete_document(document_id)
        print('[INGESTION] Starting vector indexing...')
        vector_store.upsert_chunks(chunks)
        print('[INGESTION] Vector indexing completed.')
        db.documents.update_one({'document_id': document_id}, {'$set': {'ingestion_status': 'indexed', 'updated_at': datetime.now(timezone.utc), 'ingestion_error': None}})
        return {'status': 'indexed', 'document_id': document_id, 'document_hash': document_hash, 'chunks': len(chunks), 'metadata_review_required': metadata.metadata_review_required, 'document_type': metadata.document_type, 'title': metadata.title, 'extraction_method': extraction_method}
    except Exception as e:
        db.documents.update_one({'document_id': document_id}, {'$set': {'ingestion_status': 'failed', 'ingestion_error': repr(e), 'updated_at': datetime.now(timezone.utc)}}, upsert=True)
        print('[INGESTION ERROR]', repr(e))
        raise

@stable_kb
def ingest_pdf(source_path: Path, force_ocr: bool=False, ocr_lang: str='eng', reindex: bool=False) -> dict:
    document_hash = sha256_file(source_path)
    existing = db.documents.find_one({'document_hash': document_hash})
    if existing and existing.get('ingestion_status') == 'indexed' and not reindex and not force_ocr:
        return {'status': 'duplicate', 'document_id': existing['document_id'], 'document_hash': document_hash, 'message': 'Identical PDF already exists in the knowledge base.'}
    document_id = existing['document_id'] if existing else f'DOC_{document_hash[:16].upper()}'
    stored_path = settings.upload_dir / f'{document_id}.pdf'
    if source_path.resolve() != stored_path.resolve():
        shutil.copy2(source_path, stored_path)
    db.documents.update_one({'document_id': document_id}, {'$set': {'document_id':document_id, 'document_hash':document_hash, 'stored_path':str(stored_path), 'ingestion_status':'processing', 'updated_at':datetime.now(timezone.utc)}}, upsert=True)
    pages = extract_pdf_pages(stored_path)
    print('[INGESTION] Extracted pages:', len(pages))
    use_ocr = force_ocr or needs_ocr(pages)
    if use_ocr:
        if not settings.enable_ocr:
            now = datetime.now(timezone.utc)
            db.documents.update_one({'document_id': document_id}, {'$set': {'document_id': document_id, 'document_hash': document_hash, 'filename': source_path.name, 'stored_path': str(stored_path), 'ingestion_status': 'ocr_required', 'page_count': len(pages), 'updated_at': now}, '$setOnInsert': {'created_at': now}}, upsert=True)
            return {'status': 'ocr_required', 'document_id': document_id, 'document_hash': document_hash, 'message': 'Text layer is insufficient and OCR is disabled. Enable OCR after installing Tesseract.'}
        try:
            pages = ocr_pdf_pages(stored_path, dpi=settings.ocr_dpi, lang=ocr_lang or settings.ocr_languages)
        except Exception as e:
            db.documents.update_one({'document_id': document_id}, {'$set': {'ingestion_status': 'ocr_failed', 'ocr_error': repr(e), 'updated_at': datetime.now(timezone.utc)}}, upsert=True)
            raise
        if needs_ocr(pages, min_chars_per_page=40):
            raise RuntimeError('OCR completed but extracted text is still insufficient; manual document review required.')
        return _index(document_id, document_hash, source_path, stored_path, pages, 'ocr')
    return _index(document_id, document_hash, source_path, stored_path, pages, 'text_layer')

@stable_kb
def reindex_document(document_id: str, force_ocr: bool=False, ocr_lang: str='eng') -> dict:
    d = db.documents.find_one({'document_id': document_id})
    if not d:
        raise KeyError(document_id)
    path=Path(d['stored_path'])
    path.with_suffix('.metadata.json').write_text(json.dumps(d.get('metadata',{}),ensure_ascii=False,default=str),encoding='utf-8')
    return ingest_pdf(path, force_ocr=force_ocr, ocr_lang=ocr_lang, reindex=True)
