"""Three-condition factual experiments with frozen inputs and automatic metrics."""
from __future__ import annotations
import csv
import io
import json
import random
import threading
import uuid
from collections import defaultdict
from itertools import permutations
from statistics import mean, median
from time import monotonic
from app.db.mongo import db
from app.evaluation.factual import SCORER_VERSION, score_answer
from app.evaluation.retrieval_metrics import score_retrieval
from app.services.datasets import get_questions
from app.services.factual_chat import CONDITIONS, PROMPT_VERSION, run_factual
from app.services.factual_preflight import check_sources
from app.services.operation_lock import stable_kb

_ACTIVE=set()
_ACTIVE_LOCK=threading.Lock()
RKEYS=tuple(f'{level}_hit_at_{k}' for level in ('document','chunk') for k in (1,3,5))+('document_mrr','chunk_mrr')

def _base():
    from app.services import experiments
    return experiments

def now(): return _base()._now()
def fingerprint(v): return _base()._hash(v)
def selected_questions(dataset_id, split):
    return [q for q in get_questions(dataset_id) if (q.get('factual') or {}).get('split')==split] if split else get_questions(dataset_id)

def manifest(questions, top_k, split):
    m=_base().runtime_manifest(questions,top_k)
    m['config'].update(evaluation_kind='factual',conditions=list(CONDITIONS),split=split,
        rubric_version=SCORER_VERSION,factual_prompt_version=PROMPT_VERSION,
        prompt_anchor_control='example_free_neutral_baseline',
        condition_order='deterministic six permutations by group hash',
        latency_protocol='natural cache state; startup/reloads included; order counterbalanced; no GPU warmup assumption',
        retrieval_metric_stage='final prompt evidence; pre-context/candidates exported separately',
        label_semantics=sorted({q['factual']['gold_label_kind'] for q in questions}),
        score_scope='typed factual answer only; explanations not semantically scored',
        attempt_selection='latest attempt; all attempt history retained; retries explicitly counted')
    return m

@stable_kb
def create(payload):
    from app.schemas.dataset import DatasetQuestionIn
    from app.services.datasets import validate_questions
    all_questions=get_questions(payload.dataset_id)
    validation=validate_questions([DatasetQuestionIn.model_validate(q) for q in all_questions])
    if not validation.valid: raise ValueError('; '.join(validation.errors[:20]))
    qs=selected_questions(payload.dataset_id,payload.benchmark_split)
    if not qs: raise ValueError('No factual rows in the selected split. Import a reviewed factual dataset first.')
    if payload.top_k<5: raise ValueError('Factual comparison requires top_k >= 5 to report Hit@5.')
    preflight=check_sources(qs,db)
    if not preflight['valid']: raise ValueError('; '.join(preflight['errors'][:20]))
    m=manifest(qs,payload.top_k,payload.benchmark_split)
    eid='EXP_'+uuid.uuid4().hex[:12]
    doc={'experiment_id':eid,'name':payload.name,'dataset_id':payload.dataset_id,'mode':'factual_compare',
         'evaluation_kind':'factual','benchmark_split':payload.benchmark_split,'conditions':list(CONDITIONS),
         'status':'queued','top_k':payload.top_k,'notes':payload.notes,'total_questions':len(qs),
         'completed_questions':0,'failed_questions':0,'config_snapshot':m['config'],'manifest':m,
         'source_preflight':preflight,'frozen_questions':qs,'created_at':now(),'updated_at':now()}
    db.experiments.insert_one(doc)
    return _base().get_experiment(eid)

def trace_diagnosis(q,trace):
    if trace is None: return 'retrieval_failed_before_trace'
    gold=set(q['gold_chunk_ids'])
    candidates=[c for c in trace.get('candidates',[]) if c['chunk_id'] in gold]
    if not candidates: return 'gold_not_in_dense_candidates'
    if not any(c.get('passed_threshold',False) for c in candidates): return 'gold_rejected_by_threshold'
    if not any(c['chunk_id'] in gold for c in trace.get('pre_context_results',trace.get('results',[]))):
        return 'gold_dropped_by_ranking_or_top_k'
    if not any(c['chunk_id'] in gold for c in trace.get('results',[])): return 'gold_excluded_by_context_budget'
    return 'gold_in_prompt'

def is_active(eid):
    with _ACTIVE_LOCK: return eid in _ACTIVE

@stable_kb
def run(eid):
    with _ACTIVE_LOCK:
        if eid in _ACTIVE: return
        _ACTIVE.add(eid)
    try: _run(eid)
    finally:
        with _ACTIVE_LOCK: _ACTIVE.discard(eid)

def _run(eid):
    exp=db.experiments.find_one({'experiment_id':eid},{'_id':0})
    if not exp or exp['status'] in {'completed','blocked'}: return
    # Single backend worker (embedded Qdrant). A stopped worker's running record is recoverable.
    db.experiments.update_one({'experiment_id':eid},{'$set':{'status':'running','updated_at':now()},'$unset':{'error':''}})
    try:
        qs=selected_questions(exp['dataset_id'],exp['benchmark_split'])
        if manifest(qs,exp['top_k'],exp['benchmark_split'])!=exp['manifest']:
            raise ValueError('Frozen dataset, model, KB, code or settings changed. Create a new experiment.')
        check=check_sources(qs,db)
        if not check['valid']: raise ValueError('; '.join(check['errors'][:20]))
        orders=list(permutations(CONDITIONS))
        for q in exp['frozen_questions']:
            key={'experiment_id':eid,'question_id':q['question_id']}
            item=db.experiment_items.find_one(key,{'_id':0}) or {**q,**key,'attempts':{},'scores':{},'errors':{},'timings':{},'retrieval_by_condition':{}}
            order=orders[int(fingerprint(q['group_id'])[:8],16)%len(orders)]
            item['condition_order']=list(order)
            for condition in order:
                if condition in item: continue  # resume only requests that failed, never regenerate successful answers
                started=monotonic(); answer=None;error=None;trace=None
                try:
                    answer=run_factual(q,condition,exp['top_k'])
                    trace=answer.get('retrieval_trace')
                    item[condition]=answer
                    item['errors'].pop(condition,None)
                except Exception as exc:
                    error=str(exc);trace=getattr(exc,'retrieval_trace',None)
                    item['errors'][condition]=error
                    db.experiment_errors.insert_one({**key,'system':condition,'error':error,'created_at':now()})
                elapsed=monotonic()-started
                score=score_answer(q['factual'],answer.get('answer') if answer else None,error)
                item['scores'][condition]=score;item['timings'][condition]=elapsed
                attempt={'time':now(),'seconds':elapsed,'error':error,'score':score,'answer':answer}
                item['attempts'].setdefault(condition,[]).append(attempt)
                if condition!='llm_only':
                    t=trace or {}
                    item['retrieval_by_condition'][condition]={
                        'final':score_retrieval(q['gold_document_ids'],q['gold_chunk_ids'],t.get('results',[])),
                        'candidates':score_retrieval(q['gold_document_ids'],q['gold_chunk_ids'],t.get('candidates',[])),
                        'pre_context':score_retrieval(q['gold_document_ids'],q['gold_chunk_ids'],t.get('pre_context_results',[])),
                        'trace_available':trace is not None,'diagnosis':trace_diagnosis(q,trace),
                        'trace':trace}
                db.experiment_items.update_one(key,{'$set':item},upsert=True)
                db.experiments.update_one({'experiment_id':eid},{'$set':{'updated_at':now()}})
            rows=list(db.experiment_items.find({'experiment_id':eid},{'_id':0}))
            completed=sum(all(c in r for c in CONDITIONS) for r in rows)
            failed=sum(bool(r.get('errors')) for r in rows)
            db.experiments.update_one({'experiment_id':eid},{'$set':{'completed_questions':completed,'failed_questions':failed}})
        if manifest(qs,exp['top_k'],exp['benchmark_split'])!=exp['manifest']:
            raise ValueError('Environment changed during run. Results blocked.')
        check=check_sources(qs,db)
        if not check['valid']: raise ValueError('; '.join(check['errors'][:20]))
        db.experiments.update_one({'experiment_id':eid},{'$set':{'status':'partial' if failed else 'completed','finished_at':now(),'updated_at':now()}})
    except Exception as exc:
        db.experiments.update_one({'experiment_id':eid},{'$set':{'status':'blocked','error':str(exc),'updated_at':now()}})

def avg(values):
    v=[x for x in values if x is not None]
    return mean(v) if v else None

def summarize(rows):
    out={'count':len(rows),'group_count':len({r['group_id'] for r in rows}),'systems':{}}
    for c in CONDITIONS:
        attempted=[r for r in rows if c in r.get('scores',{})]
        successful=[r for r in attempted if not r['scores'][c]['request_failed']]
        failures=len(attempted)-len(successful)
        invalid=sum(not r['scores'][c]['valid_output'] for r in successful)
        correct=sum(r['scores'][c]['correct'] for r in attempted)
        durations=[r['timings'][c] for r in attempted]
        s={'total_questions':len(rows),'attempted_count':len(attempted),'correct_count':correct,
           'factual_accuracy':correct/len(rows) if rows else None,
           'all_attempted':len(rows)==len(attempted),'not_attempted_count':len(rows)-len(attempted),
           'failed_count':failures,'failure_rate':failures/len(attempted) if attempted else None,
           'successful_response_count':len(successful),'invalid_output_count':invalid,
           'invalid_output_rate':invalid/len(successful) if successful else None,
           'mean_seconds':avg(durations),'median_seconds':median(durations) if durations else None,
           'p95_seconds':sorted(durations)[max(0,__import__('math').ceil(.95*len(durations))-1)] if durations else None,
           'retry_count':sum(max(0,len(r.get('attempts',{}).get(c,[]))-1) for r in rows),
           'total_attempt_seconds':sum(a['seconds'] for r in rows for a in r.get('attempts',{}).get(c,[])),
           'abstention_rate':avg([float(r['scores'][c]['abstained']) for r in successful if r['scores'][c]['abstained'] is not None]),
           'retrieval':None,'retrieval_denominators':None}
        if c!='llm_only':
            values=[r.get('retrieval_by_condition',{}).get(c,{}).get('final',{}) for r in attempted]
            s['retrieval']={k:avg([v.get(k) for v in values]) for k in RKEYS}
            s['retrieval_denominators']={k:sum(v.get(k) is not None for v in values) for k in RKEYS}
            s['retrieval_failure_count']=sum(not r.get('retrieval_by_condition',{}).get(c,{}).get('trace_available') for r in attempted)
        out['systems'][c]=s
    return out

def paired(rows,a,b):
    groups=defaultdict(list)
    for r in rows:
        scores=r.get('scores',{})
        if a in scores and b in scores: groups[r['group_id']].append(scores[b]['correct']-scores[a]['correct'])
    differences=[mean(v) for v in groups.values()]
    result={'baseline':a,'comparison':b,'paired_groups':len(groups),'paired_rows':sum(map(len,groups.values())),
            'mean_accuracy_delta':avg(differences),'bootstrap_95_interval':None,
            'method':'paired question-group bootstrap, 2000 resamples, seed 42; small-sample exploratory estimate'}
    if len(differences)>=2:
        rng=random.Random(42);samples=sorted(mean(rng.choices(differences,k=len(differences))) for _ in range(2000))
        result['bootstrap_95_interval']=[samples[49],samples[1949]]
    return result

def aggregate(eid):
    exp=db.experiments.find_one({'experiment_id':eid},{'_id':0})
    if not exp: raise KeyError(eid)
    saved={r['question_id']:r for r in db.experiment_items.find({'experiment_id':eid},{'_id':0})}
    rows=[saved.get(q['question_id'],q) for q in exp['frozen_questions']]
    result=summarize(rows)
    result.update(evaluation_kind='factual',scorer_version=SCORER_VERSION,conditions=list(CONDITIONS),
        score_scope='typed factual answer only; legal explanations not evaluated',
        question_count=len(rows),expected_question_count=exp['total_questions'],
        label_semantics=exp['config_snapshot']['label_semantics'],
        attempt_selection='latest attempt; failures/invalid/unattempted answers are not correct; inspect retries',
        valid_for_reporting=exp['status']=='completed')
    for field in ('language','domain'):
        groups=defaultdict(list)
        for r in rows:groups[r[field]].append(r)
        result['by_'+field]={k:summarize(v) for k,v in groups.items()}
    result['paired_comparisons']=[paired(rows,*p) for p in [('llm_only','rag_no_rerank'),('llm_only','rag_rerank'),('rag_no_rerank','rag_rerank')]]
    for lang,value in result['by_language'].items():
        value['paired_comparisons']=[paired([r for r in rows if r['language']==lang],*p) for p in [('llm_only','rag_no_rerank'),('llm_only','rag_rerank')]]
    return result

def export_csv(eid, summary=False):
    exp=db.experiments.find_one({'experiment_id':eid},{'_id':0})
    if not exp:raise KeyError(eid)
    buf=io.StringIO()
    if summary:
        aggregate_data=aggregate(eid)
        fields=['breakdown','label','condition','rows','groups','accuracy','hit_at_1','hit_at_3','hit_at_5','mrr','median_seconds','p95_seconds','failure_rate','invalid_output_rate','retries','valid_for_reporting','label_semantics']
        w=csv.DictWriter(buf,fieldnames=fields);w.writeheader()
        sections=[('overall','all',aggregate_data)]
        for dimension in ('language','domain'):
            sections.extend((dimension,label,data) for label,data in aggregate_data['by_'+dimension].items())
        for dimension,label,data in sections:
            for c,s in data['systems'].items():
                ret=s['retrieval'] or {}
                w.writerow(dict(breakdown=dimension,label=label,condition=c,rows=data['count'],groups=data['group_count'],
                    accuracy=s['factual_accuracy'],hit_at_1=ret.get('chunk_hit_at_1'),hit_at_3=ret.get('chunk_hit_at_3'),hit_at_5=ret.get('chunk_hit_at_5'),
                    mrr=ret.get('chunk_mrr'),median_seconds=s['median_seconds'],p95_seconds=s['p95_seconds'],failure_rate=s['failure_rate'],
                    invalid_output_rate=s['invalid_output_rate'],retries=s['retry_count'],valid_for_reporting=aggregate_data['valid_for_reporting'],label_semantics=';'.join(aggregate_data['label_semantics'])))
    else:
        fields=['question_id','group_id','language','domain','question','condition','factual','reference_answer','gold_document_ids','gold_chunk_ids','answer','score','seconds','error','retrieval','attempts']
        w=csv.DictWriter(buf,fieldnames=fields);w.writeheader()
        for item in db.experiment_items.find({'experiment_id':eid},{'_id':0}).sort('question_id',1):
            for c in CONDITIONS:
                row={k:item.get(k) for k in fields};row.update(condition=c,answer=item.get(c,{}).get('answer'),score=item.get('scores',{}).get(c),
                    seconds=item.get('timings',{}).get(c),error=item.get('errors',{}).get(c),retrieval=item.get('retrieval_by_condition',{}).get(c),attempts=item.get('attempts',{}).get(c,[]))
                w.writerow({k:json.dumps(v,ensure_ascii=False,default=str) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    return buf.getvalue().encode('utf-8-sig')
