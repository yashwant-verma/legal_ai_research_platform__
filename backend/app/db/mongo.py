from pymongo import ASCENDING, DESCENDING, MongoClient
from app.core.config import get_settings
settings=get_settings(); client=MongoClient(settings.mongodb_uri); db=client[settings.mongodb_db]
def ensure_indexes()->None:
    db.documents.create_index([('document_hash',ASCENDING)],unique=True); db.documents.create_index([('document_id',ASCENDING)],unique=True)
    for f in ('metadata.document_type','metadata.jurisdiction','metadata.status','metadata.effective_from','metadata.effective_to'): db.documents.create_index([(f,ASCENDING)])
    db.chunks.create_index([('chunk_id',ASCENDING)],unique=True); db.chunks.create_index([('document_id',ASCENDING)]); db.chunks.create_index([('section',ASCENDING)]); db.chunks.create_index([('article',ASCENDING)])
    db.research_runs.create_index([('run_id',ASCENDING)],unique=True); db.research_runs.create_index([('created_at',DESCENDING)])
    db.datasets.create_index([('dataset_id',ASCENDING)],unique=True); db.datasets.create_index([('name',ASCENDING),('version',ASCENDING)])
    db.dataset_questions.create_index([('dataset_id',ASCENDING),('question_id',ASCENDING)],unique=True); db.dataset_questions.create_index([('group_id',ASCENDING)]); db.dataset_questions.create_index([('language',ASCENDING)]); db.dataset_questions.create_index([('domain',ASCENDING)])
    db.experiments.create_index([('experiment_id',ASCENDING)],unique=True); db.experiments.create_index([('created_at',DESCENDING)])
    db.experiment_items.create_index([('experiment_id',ASCENDING),('question_id',ASCENDING)],unique=True)
    old_key=[('experiment_id',ASCENDING),('question_id',ASCENDING),('system',ASCENDING)]
    for name,info in db.evaluations.index_information().items():
        if info.get('key')==old_key: db.evaluations.drop_index(name)
    db.evaluations.create_index(old_key+[('evaluator',ASCENDING)],unique=True)
    db.experiment_errors.create_index([('experiment_id',ASCENDING)])

    # Single-worker startup recovery: interrupted work is explicit and retryable.
    db.experiments.update_many({'status':'running'},{'$set':{'status':'partial','error':'Server restarted. Resume with unchanged environment or create a new experiment.'}})
    db.ingestion_jobs.update_many({'status':{'$in':['queued','processing']}},{'$set':{'status':'failed','error':'Server restarted during indexing. Reindex or upload again.'}})
    db.documents.update_many({'ingestion_status':{'$in':['processing','indexing']}},{'$set':{'ingestion_status':'failed','ingestion_error':'Interrupted indexing; reindex required.'}})
