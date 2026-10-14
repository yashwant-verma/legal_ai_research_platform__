import json
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.schemas.dataset import DatasetCreate
from app.services.datasets import create_dataset, delete_dataset, get_dataset, get_questions, list_datasets, parse_dataset_bytes, rows_to_questions, validate_questions
router=APIRouter(prefix='/datasets',tags=['datasets'])

@router.get('')
def all_datasets(): return list_datasets()
@router.post('')
def new_dataset(payload:DatasetCreate):
    try: return create_dataset(payload)
    except ValueError as e: raise HTTPException(422,str(e))
@router.get('/{dataset_id}')
def one(dataset_id:str):
    try: return get_dataset(dataset_id)
    except KeyError: raise HTTPException(404,'Dataset not found')
@router.get('/{dataset_id}/questions')
def questions(dataset_id:str): return get_questions(dataset_id)
@router.get('/{dataset_id}/validate')
def validate(dataset_id:str):
    qs=[__import__('app.schemas.dataset',fromlist=['DatasetQuestionIn']).DatasetQuestionIn.model_validate(x) for x in get_questions(dataset_id)]
    return validate_questions(qs)
@router.delete('/{dataset_id}',status_code=204)
def remove(dataset_id:str):
    try: delete_dataset(dataset_id)
    except ValueError as e: raise HTTPException(409,str(e))
@router.post('/import')
async def import_dataset(file:UploadFile=File(...), name:str=Form(...), version:str=Form('v1.0'), benchmark:bool=Form(False), mapping_json:str=Form('{}')):
    try:
        rows=parse_dataset_bytes(file.filename or 'dataset.csv',await file.read()); mapping=json.loads(mapping_json or '{}'); qs=rows_to_questions(rows,mapping)
        return create_dataset(DatasetCreate(name=name,version=version,benchmark=benchmark,questions=qs))
    except Exception as e: raise HTTPException(422,str(e))

@router.get('/{dataset_id}/factual-preflight')
def factual_preflight(dataset_id:str):
    from app.db.mongo import db
    from app.services.factual_preflight import check_sources
    from app.services.operation_lock import stable_kb
    try: return stable_kb(check_sources)(get_questions(dataset_id),db)
    except Exception as exc: raise HTTPException(422,str(exc))
