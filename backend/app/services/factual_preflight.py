import hashlib
from app.schemas.dataset import DatasetQuestionIn
from app.services.datasets import validate_questions
from app.schemas.factual import DOMAINS

def check_sources(questions: list[dict], database, check_vectors=True) -> dict:
    rows=[DatasetQuestionIn.model_validate(q) for q in questions]
    validation=validate_questions(rows)
    errors=list(validation.errors)
    chunks={c['chunk_id']:c for c in database.chunks.find({}, {'_id':0})}
    docs={d['document_id']:d for d in database.documents.find({}, {'_id':0})}
    active_acts={}
    for did,d in docs.items():
        m=d.get('metadata',{})
        if d.get('ingestion_status')!='indexed' or m.get('status') in {'historical','repealed'}: continue
        if not m.get('act_id') or not m.get('version') or m.get('domain') not in DOMAINS or m.get('language')!='en' or m.get('metadata_review_required',True):
            errors.append(f'{did}: active benchmark corpus needs reviewed Act/domain/version and English metadata.')
        if m.get('status') not in {'in_force','amended'}:
            errors.append(f'{did}: verify the legal status before a factual benchmark.')
        aid=m.get('act_id')
        if aid in active_acts:
            errors.append(f'{aid}: multiple active source documents/editions; keep one benchmark edition and mark old versions historical.')
        active_acts[aid]=did
    selected={}
    for q in rows:
        if not q.factual:
            errors.append(f'{q.question_id}: requires factual labels.'); continue
        f=q.factual
        if f.review_status!='verified': errors.append(f'{q.question_id}: review the factual group before experiments.')
        for did in q.gold_document_ids:
            if did not in docs or docs[did].get('ingestion_status')!='indexed': errors.append(f'{did}: not indexed.')
        for cid in q.gold_chunk_ids:
            c=chunks.get(cid)
            if not c:
                errors.append(f'{cid}: missing from Mongo.');continue
            selected[cid]=c
            d=docs.get(c['document_id'],{}); m=d.get('metadata',{})
            if c.get('metadata')!=m: errors.append(f'{cid}: document/chunk metadata mismatch; reindex.')
            if c['document_id'] not in q.gold_document_ids: errors.append(f'{cid}: gold document ownership mismatch.')
            if d.get('ingestion_status')!='indexed': errors.append(f'{cid}: source document not indexed.')
            if hashlib.sha256(c['text'].encode()).hexdigest()!=q.gold_chunk_hashes.get(cid): errors.append(f'{cid}: source text changed.')
            if m.get('act_id')!=f.act_id or m.get('version')!=f.source_version: errors.append(f'{cid}: Act/version differs from reviewed labels.')
            if m.get('domain')!=q.domain: errors.append(f'{cid}: domain differs from source metadata.')
            if m.get('metadata_review_required',True): errors.append(f'{cid}: source metadata needs review.')
            if m.get('language')!='en': errors.append(f'{cid}: factual v1 benchmark requires English source corpus.')
            if m.get('status') not in {'in_force','amended'}: errors.append(f'{cid}: source status is outside the current-law benchmark.')
            for quote in f.supporting_quotes:
                if quote.chunk_id==cid and quote.quote not in c['text']: errors.append(f'{cid}: supporting quote not found verbatim.')
    vectors_checked=False
    if not errors and check_vectors:
        from app.services.vector_store import get_vector_store
        # Verify the entire indexed corpus: stale negatives can change rankings too.
        indexed=[c for c in chunks.values() if docs.get(c['document_id'],{}).get('ingestion_status')=='indexed']
        errors.extend(get_vector_store().verify_chunks(indexed))
        vectors_checked=True
    return {'valid':not errors,'errors':errors,'warnings':validation.warnings,
            'checked_gold_chunks':len(selected),'vector_check_performed':vectors_checked}
