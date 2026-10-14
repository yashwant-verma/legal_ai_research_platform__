import json
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from app.db.mongo import db
from app.models.legal import LegalMetadata
from app.services.ingestion import ingest_pdf, reindex_document
from app.services.operation_lock import stable_kb
router = APIRouter(prefix='/kb', tags=['knowledge-base'])

@router.get('/scope')
def legal_scope():
    return json.loads((Path(__file__).resolve().parents[3]/'config/legal_scope.json').read_text(encoding='utf-8'))

def _run_ingestion_job(job_id: str, pdf_path: str, temp_dir: str, force_ocr: bool, ocr_lang: str):
    """
    Run the existing ingest_pdf() pipeline separately from
    the HTTP request.

    This prevents frontend request timeout while extraction,
    chunking, embeddings and vector indexing continue.
    """
    print(f'[INGESTION JOB] Started: {job_id}')
    path = Path(pdf_path)
    try:
        db.ingestion_jobs.update_one({'job_id': job_id}, {'$set': {'status': 'processing', 'progress': 10, 'started_at': datetime.now(timezone.utc), 'updated_at': datetime.now(timezone.utc), 'error': None}})
        result = ingest_pdf(path, force_ocr=force_ocr, ocr_lang=ocr_lang)
        db.ingestion_jobs.update_one({'job_id': job_id}, {'$set': {'status': 'completed', 'progress': 100, 'result': jsonable_encoder(result), 'completed_at': datetime.now(timezone.utc), 'updated_at': datetime.now(timezone.utc), 'error': None}})
        print(f'[INGESTION JOB] Completed: {job_id}')
    except Exception as e:
        db.ingestion_jobs.update_one({'job_id': job_id}, {'$set': {'status': 'failed', 'error': repr(e), 'completed_at': datetime.now(timezone.utc), 'updated_at': datetime.now(timezone.utc)}})
        print(f'[INGESTION JOB ERROR] {job_id}: {repr(e)}')
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_error:
            print('[INGESTION CLEANUP ERROR]', repr(cleanup_error))

@router.post('/ingest', status_code=202)
def ingest(file: UploadFile=File(...), metadata_json: str | None=Form(None), force_ocr: bool=Form(False), ocr_lang: str=Form('eng')):
    print('[INGESTION] Upload request received')
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF files are accepted')
    metadata_data = None
    if metadata_json:
        try:
            metadata_data = json.loads(metadata_json)
            LegalMetadata.model_validate(metadata_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Invalid legal metadata: {e}')
    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix='legal_ingest_')
    try:
        safe_filename = Path(file.filename).name
        pdf_path = Path(temp_dir) / safe_filename
        with pdf_path.open('wb') as out:
            shutil.copyfileobj(file.file, out)
        if metadata_data is not None:
            metadata_path = pdf_path.with_suffix('.metadata.json')
            metadata_path.write_text(json.dumps(metadata_data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f'Failed to save uploaded PDF: {e}')
    now = datetime.now(timezone.utc)
    try:
        db.ingestion_jobs.insert_one({'job_id': job_id, 'filename': file.filename, 'status': 'queued', 'progress': 0, 'force_ocr': force_ocr, 'ocr_lang': ocr_lang, 'result': None, 'error': None, 'created_at': now, 'updated_at': now})
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f'Could not create ingestion job: {e}')
    worker = threading.Thread(target=_run_ingestion_job, args=(job_id, str(pdf_path), temp_dir, force_ocr, ocr_lang), daemon=True)
    worker.start()
    print(f'[INGESTION] Job queued: {job_id}')
    return {'job_id': job_id, 'filename': file.filename, 'status': 'queued', 'progress': 0, 'message': 'PDF ingestion started in background.'}

@router.get('/ingest/jobs/{job_id}')
def ingestion_job(job_id: str):
    job = db.ingestion_jobs.find_one({'job_id': job_id}, {'_id': 0})
    if not job:
        raise HTTPException(status_code=404, detail='Ingestion job not found')
    return job

@router.get('/ingest/jobs')
def ingestion_jobs(limit: int=20):
    safe_limit = min(max(limit, 1), 100)
    return list(db.ingestion_jobs.find({}, {'_id': 0}).sort('created_at', -1).limit(safe_limit))

@router.get('/documents')
def documents(status: str | None=None, review_required: bool | None=None, limit: int=100, skip: int=0):
    q = {}
    if status:
        q['ingestion_status'] = status
    if review_required is not None:
        q['metadata.metadata_review_required'] = review_required
    return list(db.documents.find(q, {'_id': 0}).sort('created_at', -1).skip(skip).limit(min(limit, 500)))

@router.get('/documents/{document_id}')
def document(document_id: str):
    d = db.documents.find_one({'document_id': document_id}, {'_id': 0})
    if not d:
        raise HTTPException(status_code=404, detail='Document not found')
    d['chunks'] = list(db.chunks.find({'document_id': document_id}, {'_id': 0, 'text': 0}).sort('page_start', 1))
    return d

@router.put('/documents/{document_id}/metadata')
@stable_kb
def update_metadata(document_id: str, metadata: LegalMetadata):
    if not db.documents.find_one({'document_id': document_id}):
        raise HTTPException(status_code=404, detail='Document not found')
    metadata_dump = metadata.model_dump(mode='json')
    db.documents.update_one({'document_id': document_id}, {'$set': {'metadata': metadata_dump, 'ingestion_status':'needs_reindex'}})
    db.chunks.update_many({'document_id': document_id}, {'$set': {'metadata': metadata_dump}})
    return {'document_id': document_id, 'metadata_updated': True, 'warning': 'Reindex the document so vector-store payload metadata also matches.'}

def _run_reindex(job_id, document_id, force_ocr, ocr_lang):
    db.ingestion_jobs.update_one({'job_id':job_id},{'$set':{'status':'processing'}})
    try:
        result=reindex_document(document_id, force_ocr, ocr_lang)
        db.ingestion_jobs.update_one({'job_id':job_id},{'$set':{'status':'completed','result':result,'updated_at':datetime.now(timezone.utc)}})
    except Exception as exc:
        db.ingestion_jobs.update_one({'job_id':job_id},{'$set':{'status':'failed','error':str(exc),'updated_at':datetime.now(timezone.utc)}})

@router.post('/documents/{document_id}/reindex',status_code=202)
def reindex(document_id: str, force_ocr: bool=False, ocr_lang: str='eng'):
    if not db.documents.find_one({'document_id':document_id}): raise HTTPException(404,'Document not found')
    job={'job_id':str(uuid.uuid4()),'document_id':document_id,'filename':document_id,'status':'queued','created_at':datetime.now(timezone.utc),'error':None}
    db.ingestion_jobs.insert_one(dict(job))
    threading.Thread(target=_run_reindex,args=(job['job_id'],document_id,force_ocr,ocr_lang),daemon=True).start()
    return job

@router.get('/stats')
def stats():
    statuses = {x['_id']: x['count'] for x in db.documents.aggregate([{'$group': {'_id': '$ingestion_status', 'count': {'$sum': 1}}}])}
    return {'documents': db.documents.count_documents({}), 'indexed': statuses.get('indexed', 0), 'failed': statuses.get('failed', 0) + statuses.get('ocr_failed', 0), 'ocr_required': statuses.get('ocr_required', 0), 'chunks': db.chunks.count_documents({}), 'metadata_review_required': db.documents.count_documents({'metadata.metadata_review_required': True}), 'statuses': statuses}
