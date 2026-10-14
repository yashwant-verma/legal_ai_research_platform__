from __future__ import annotations
import csv, io, json, uuid, hashlib, random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
import httpx
from importlib.metadata import version, PackageNotFoundError
from app.core.config import get_settings
from app.db.mongo import db
from app.evaluation.retrieval_metrics import score_retrieval
from app.evaluation.citations import citation_reference_validity
from app.schemas.experiment import ExperimentCreate, ManualEvaluationIn
from app.schemas.retrieval import RetrievalFilters
from app.services.chat import run_llm_only, run_rag
from app.services.datasets import get_questions
from app.services.operation_lock import stable_kb

settings=get_settings()
METRICS=('correctness','groundedness','citation_correctness','hallucination_rate')
RKEYS=tuple(f'{level}_hit_at_{k}' for level in ('document','chunk') for k in (1,3,5))+('document_mrr','chunk_mrr')
def _now(): return datetime.now(timezone.utc)
def _hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,ensure_ascii=False).encode()).hexdigest()
def _avg(values):
    vals=[v for v in values if v is not None]
    return sum(vals)/len(vals) if vals else None

def config_snapshot(top_k:int)->dict:
    keys=('llm_provider','ollama_think','llm_max_output_tokens','prompt_version','embedding_model','embedding_device','reranker_device','reranker_model','enable_reranker','dense_min_score','retrieval_candidate_k','rerank_top_n','chunk_version','qdrant_collection','ollama_num_ctx','ollama_num_gpu','rag_context_max_chars','ollama_timeout_seconds','ollama_max_duration_seconds')
    return {**{k:getattr(settings,k) for k in keys},'model':settings.ollama_model,'temperature':settings.llm_temperature,'top_k':top_k,'retrieval_metric_stage':'final_evidence','document_ranking':'first_occurrence_unique_documents','rubric_version':'legal_review_v2'}

def source_snapshot():
    docs=list(db.documents.find({}, {'_id':0,'document_id':1,'document_hash':1,'ingestion_status':1,'chunk_version':1,'embedding_model':1,'metadata':1}))
    if any(d.get('ingestion_status') in {'indexing','processing'} for d in docs):
        raise ValueError('Knowledge base is changing. Finish ingestion before creating/running an experiment.')
    chunks=list(db.chunks.find({}, {'_id':0}))
    return _hash({'documents':sorted(docs,key=lambda x:x['document_id']),'chunks':sorted(chunks,key=lambda x:x['chunk_id'])})

def model_digest():
    # Ollama tags expose the installed model digest without loading the model.
    with httpx.Client(timeout=10,trust_env=False) as client:
        r=client.get(settings.ollama_base_url.rstrip('/')+'/api/tags');r.raise_for_status()
    name=settings.ollama_model
    model=next((m for m in r.json().get('models',[]) if m.get('name') in {name,name+':latest'} or m.get('model') in {name,name+':latest'}),None)
    if not model or not model.get('digest'): raise ValueError('Configured Ollama model is not installed or has no digest.')
    return model['digest']

def runtime_manifest(questions, top_k):
    files=Path(__file__).parents[1].rglob('*.py')
    code=_hash({str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    dependencies={}
    for name in ('fastapi','pydantic','pymongo','qdrant-client','sentence-transformers','torch','httpx'):
        try: dependencies[name]=version(name)
        except PackageNotFoundError: dependencies[name]='not-installed'
    return {'dependencies':dependencies,'config':config_snapshot(top_k),'dataset_hash':_hash(questions),'kb_hash':source_snapshot(),'model_digest':model_digest(),'code_hash':code}

@stable_kb
def create_experiment(payload:ExperimentCreate)->dict:
    if payload.mode == 'factual_compare':
        from app.services.factual_experiments import create
        return create(payload)
    questions=get_questions(payload.dataset_id)
    if not questions: raise ValueError('Dataset does not exist or contains no questions.')
    # Validate source ownership and current IDs before any expensive model requests.
    live={c['chunk_id']:c for c in db.chunks.find({}, {'_id':0,'chunk_id':1,'document_id':1,'text':1})}
    docs={d['document_id'] for d in db.documents.find({'ingestion_status':'indexed'},{'document_id':1})}
    for q in questions:
        for cid in q.get('gold_chunk_ids',[]):
            if cid not in live or live[cid]['document_id'] not in docs or (q.get('gold_document_ids') and live[cid]['document_id'] not in q['gold_document_ids']):
                raise ValueError(f"{q['question_id']}: gold chunk missing, not indexed, or belongs to another document; regenerate/review labels.")
        for cid,expected in q.get('gold_chunk_hashes',{}).items():
            if cid not in live or hashlib.sha256(live[cid]['text'].encode('utf-8')).hexdigest()!=expected:
                raise ValueError(f"{q['question_id']}: gold source text changed; re-review benchmark.")
        if set(q.get('gold_document_ids',[]))-docs: raise ValueError(f"{q['question_id']}: gold document is not indexed.")
    manifest=runtime_manifest(questions,payload.top_k)
    eid=f"EXP_{uuid.uuid4().hex[:12]}"
    d={'experiment_id':eid,'name':payload.name,'dataset_id':payload.dataset_id,'mode':payload.mode,'status':'queued','top_k':payload.top_k,'notes':payload.notes,'total_questions':len(questions),'completed_questions':0,'failed_questions':0,'config_snapshot':manifest['config'],'manifest':manifest,'created_at':_now(),'updated_at':_now()}
    db.experiments.insert_one(d)
    return get_experiment(eid)

def get_experiment(eid):
    d=db.experiments.find_one({'experiment_id':eid},{'_id':0})
    if not d: raise KeyError(eid)
    d['aggregate']=aggregate_experiment(eid) if d['status'] in {'completed','partial','failed','blocked'} else None
    return d

def list_experiments():
    return [get_experiment(x['experiment_id']) for x in db.experiments.find({}, {'experiment_id':1,'_id':0}).sort('created_at',-1)]

@stable_kb
def run_experiment(eid):
    existing=db.experiments.find_one({'experiment_id':eid},{'_id':0})
    if existing and existing.get('evaluation_kind')=='factual':
        from app.services.factual_experiments import run
        return run(eid)
    # Atomic claim makes duplicate HTTP run requests harmless.
    claim=db.experiments.update_one({'experiment_id':eid,'status':{'$in':['queued','partial','failed']}},{'$set':{'status':'running','started_at':_now()}})
    if not claim.modified_count: return
    exp=db.experiments.find_one({'experiment_id':eid})
    completed=failed=0
    try:
        questions=get_questions(exp['dataset_id'])
        if not exp.get('manifest') or runtime_manifest(questions,exp['top_k']) != exp['manifest']:
            raise ValueError('Dataset, KB, installed model, code or settings changed. Create a NEW experiment; previous results are preserved.')
        for q in questions:
            item=db.experiment_items.find_one({'experiment_id':eid,'question_id':q['question_id']},{'_id':0}) or {
                **q,'experiment_id':eid,'created_at':_now(),'errors':{},'timings':{}}
            item.setdefault('errors',{});item.setdefault('timings',{})
            systems=['llm_only','rag'] if exp['mode']=='compare' else [exp['mode']]
            # Alternate order by group to reduce systematic model warm-up advantage.
            if int(_hash(q.get('group_id') or q['question_id'])[:8],16)%2: systems.reverse()
            for system in systems:
                if system in item: continue
                started=monotonic()
                try:
                    ans=run_llm_only(q['question']) if system=='llm_only' else run_rag(q['question'],RetrievalFilters(),top_k=exp['top_k'])
                    item[system]=ans.model_dump(mode='json');item['errors'].pop(system,None)
                    if system=='rag':
                        ev=item['rag']['evidence']
                        gold_docs=q.get('gold_document_ids',[]) if q.get('answerable_from_kb',True) else []
                        gold_chunks=q.get('gold_chunk_ids',[]) if q.get('answerable_from_kb',True) else []
                        item['retrieval_metrics']=score_retrieval(gold_docs,gold_chunks,ev)
                        trace=item['rag'].get('retrieval_trace') or {}
                        item['candidate_retrieval_metrics']=score_retrieval(gold_docs,gold_chunks,trace.get('candidates',[]))
                        item['retrieval_status']='insufficient_evidence' if ans.abstained else 'evidence_found'
                        item['citation_syntax_metrics']=citation_reference_validity(ans.answer,len(ev))
                except Exception as exc:
                    item['errors'][system]=str(exc)
                    trace=getattr(exc,'retrieval_trace',None)
                    if trace is not None:
                        item['failed_retrieval_trace']=trace
                        gd=q.get('gold_document_ids',[]) if q.get('answerable_from_kb',True) else []
                        gc=q.get('gold_chunk_ids',[]) if q.get('answerable_from_kb',True) else []
                        item['retrieval_metrics']=score_retrieval(gd,gc,trace.get('results',[]))
                        item['candidate_retrieval_metrics']=score_retrieval(gd,gc,trace.get('candidates',[]))
                    db.experiment_errors.insert_one({'experiment_id':eid,'question_id':q['question_id'],'system':system,'error':str(exc),'created_at':_now()})
                finally:
                    item['timings'][system]=monotonic()-started
                    db.experiment_items.update_one({'experiment_id':eid,'question_id':q['question_id']},{'$set':item},upsert=True)
            if all(s in item for s in systems): completed+=1
            else: failed+=1
            db.experiments.update_one({'experiment_id':eid},{'$set':{'completed_questions':completed,'failed_questions':failed,'updated_at':_now()}})
        if runtime_manifest(questions,exp['top_k']) != exp['manifest']:
            raise ValueError('Environment changed during execution. Results are blocked; create a new experiment.')
        status='completed' if not failed else ('partial' if completed else 'failed')
        db.experiments.update_one({'experiment_id':eid},{'$set':{'status':status,'finished_at':_now(),'updated_at':_now()}})
    except Exception as exc:
        db.experiments.update_one({'experiment_id':eid},{'$set':{'status':'blocked','error':str(exc),'updated_at':_now()}})

def save_manual_evaluation(eid,qid,system,payload):
    if system not in {'llm_only','rag'}: raise ValueError('Unknown system')
    item=db.experiment_items.find_one({'experiment_id':eid,'question_id':qid})
    if not item or system not in item: raise ValueError('Cannot evaluate a missing or failed answer.')
    evaluator=(payload.evaluator or '').strip()
    if not evaluator: raise ValueError('Evaluator name is required.')
    record={'experiment_id':eid,'question_id':qid,'system':system,**payload.model_dump(),'evaluator':evaluator,'updated_at':_now(),'rubric_version':'legal_review_v2'}
    key={k:record[k] for k in ('experiment_id','question_id','system','evaluator')}
    db.evaluations.update_one(key,{'$set':record},upsert=True)
    return db.evaluations.find_one(key,{'_id':0})

def evaluation_map(evals):
    grouped=defaultdict(list)
    for e in evals: grouped[(e['question_id'],e['system'])].append(e)
    return {key:{m:_avg([e.get(m) for e in es]) for m in METRICS} for key,es in grouped.items()}

def summarize(rows, em):
    out={'count':len(rows),'retrieval':{},'retrieval_denominators':{},'systems':{}}
    for key in RKEYS:
        vals=[r.get('retrieval_metrics',{}).get(key) for r in rows]
        out['retrieval'][key]=_avg(vals);out['retrieval_denominators'][key]=sum(v is not None for v in vals)
    for s in ('llm_only','rag'):
        scores=[em[(r['question_id'],s)] for r in rows if (r['question_id'],s) in em]
        out['systems'][s]={m:_avg([x[m] for x in scores]) for m in METRICS}
        out['systems'][s].update({'evaluated_count':len(scores),'metric_denominators':{m:sum(x[m] is not None for x in scores) for m in METRICS},'answered_count':sum(s in r for r in rows),'failed_count':sum(s in r.get('errors',{}) for r in rows),'mean_seconds':_avg([r.get('timings',{}).get(s) for r in rows if s in r])})
    # Old frontend field remains available.
    out['document_hit_at_5']=out['retrieval']['document_hit_at_5']
    return out

def paired_summary(rows, em):
    groups=defaultdict(list)
    for r in rows:
        a=em.get((r['question_id'],'llm_only'),{}).get('correctness');b=em.get((r['question_id'],'rag'),{}).get('correctness')
        if a is not None and b is not None: groups[r.get('group_id') or r['question_id']].append(b-a)
    differences=[_avg(v) for v in groups.values()]
    result={'paired_groups':len(differences),'paired_rows':sum(map(len,groups.values())),'mean_group_correctness_delta':_avg(differences),'bootstrap_95_interval':None,'method':'paired group bootstrap; 2000 resamples, seed 42; positive means RAG higher; exploratory with small samples'}
    if len(differences)>=2:
        rng=random.Random(42); samples=sorted(_avg(rng.choices(differences,k=len(differences))) for _ in range(2000))
        result['bootstrap_95_interval']=[samples[49],samples[1949]]
    return result

def aggregate_experiment(eid):
    exp=db.experiments.find_one({'experiment_id':eid},{'_id':0}) or {}
    if exp.get('evaluation_kind')=='factual':
        from app.services.factual_experiments import aggregate
        return aggregate(eid)
    rows=get_items(eid);evals=list(db.evaluations.find({'experiment_id':eid},{'_id':0}));em=evaluation_map(evals)
    out=summarize(rows,em);out['question_count']=len(rows)
    exp=db.experiments.find_one({'experiment_id':eid},{'_id':0}) or {}
    out['expected_question_count']=exp.get('total_questions',len(rows))
    out['retrieval_metric_stage']='final evidence; candidates reported separately in full JSON'
    for field in ('language','domain'):
        groups=defaultdict(list)
        for r in rows: groups[r.get(field,'unknown')].append(r)
        out['by_'+field]={k:{**summarize(v,em),'paired_comparison':paired_summary(v,em)} for k,v in groups.items()}
    out['paired_comparison']=paired_summary(rows,em)
    out['rq3']={}
    for success in (0.0,1.0):
        relevant=[r for r in rows if r.get('retrieval_metrics',{}).get('chunk_hit_at_1')==success]
        out['rq3']['hit_at_1' if success else 'miss_at_1']=summarize(relevant,em)
    rag=[r for r in rows if 'rag' in r]
    out['rag_abstention_rate']=_avg([float(r['rag'].get('abstained',False)) for r in rag])
    out['unanswerable_abstention_accuracy']=_avg([float(r['rag'].get('abstained',False)) for r in rag if r.get('answerable_from_kb') is False])
    out['answerable_false_abstention_rate']=_avg([float(r['rag'].get('abstained',False)) for r in rag if r.get('answerable_from_kb') is True])
    out['evaluation_records']=len(evals)
    return out

def get_items(eid): return list(db.experiment_items.find({'experiment_id':eid},{'_id':0}).sort('question_id',1))
def export_bundle(eid):
    experiment=get_experiment(eid)
    return {'schema_version':'factual_export_v1' if experiment.get('evaluation_kind')=='factual' else 'research_export_v2','experiment':experiment,'items':get_items(eid),'evaluations':list(db.evaluations.find({'experiment_id':eid},{'_id':0})),'errors':list(db.experiment_errors.find({'experiment_id':eid},{'_id':0}))}

def export_csv(eid):
    exp=db.experiments.find_one({'experiment_id':eid},{'_id':0}) or {}
    if exp.get('evaluation_kind')=='factual':
        from app.services.factual_experiments import export_csv as factual_csv
        return factual_csv(eid)
    bundle=export_bundle(eid);em=evaluation_map(bundle['evaluations']);buf=io.StringIO()
    fields=['question_id','group_id','language','domain','question','reference_answer','answerable_from_kb','gold_document_ids','gold_chunk_ids','llm_only_answer','rag_answer','rag_abstained',*RKEYS,*[s+'_'+m for s in ('llm_only','rag') for m in METRICS],'timings','errors','retrieval_trace','manifest']
    writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader()
    for r in bundle['items']:
        row={k:r.get(k) for k in fields}
        row.update({s+'_answer':r.get(s,{}).get('answer') for s in ('llm_only','rag')});row['rag_abstained']=r.get('rag',{}).get('abstained')
        row.update({k:r.get('retrieval_metrics',{}).get(k) for k in RKEYS})
        row.update({s+'_'+m:em.get((r['question_id'],s),{}).get(m) for s in ('llm_only','rag') for m in METRICS})
        row['retrieval_trace']=r.get('rag',{}).get('retrieval_trace');row['manifest']=bundle['experiment'].get('manifest')
        writer.writerow({k:json.dumps(v,ensure_ascii=False,default=str) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return buf.getvalue().encode('utf-8-sig')
