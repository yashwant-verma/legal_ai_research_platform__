import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import mongomock
from app.services import experiments as e
from app.schemas.chat import ChatAnswer
from app.schemas.experiment import ManualEvaluationIn,ExperimentCreate
from app.evaluation.language_checks import language_issues
from app.services.llm.output_filter import visible_tokens
from app.services.llm.base import LLMError
from app.models.legal import PageText,LegalMetadata
from app.services.chunking import chunk_document
from scripts.calibrate_retrieval import sweep

@pytest.fixture
def memory(monkeypatch):
    db=mongomock.MongoClient().db
    monkeypatch.setattr(e,'db',db)
    monkeypatch.setattr(e,'get_questions',lambda ds:list(db.dataset_questions.find({'dataset_id':ds},{'_id':0})))
    monkeypatch.setattr(e,'runtime_manifest',lambda qs,k:{'config':e.config_snapshot(k),'fixed':'snapshot'})
    return db

def answer(system):return ChatAnswer(system_type=system,answer='Complete answer',prompt_version='v2',model='local')

def test_branch_failure_preserved_retry_and_manifest_block(memory,monkeypatch):
    memory.dataset_questions.insert_one({'dataset_id':'D','question_id':'Q','group_id':'G','language':'en','domain':'law','question':'Question?', 'answerable_from_kb':False})
    exp=e.create_experiment(ExperimentCreate(name='Test',dataset_id='D'));eid=exp['experiment_id']
    baseline=Mock(return_value=answer('llm_only'));rag=Mock(side_effect=RuntimeError('GPU failed'))
    monkeypatch.setattr(e,'run_llm_only',baseline);monkeypatch.setattr(e,'run_rag',rag)
    e.run_experiment(eid)
    item=e.get_items(eid)[0]
    assert item['llm_only']['answer']=='Complete answer' and 'GPU failed' in item['errors']['rag']
    assert e.get_experiment(eid)['failed_questions']==1
    rag.side_effect=None;rag.return_value=answer('rag')
    e.run_experiment(eid)
    assert baseline.call_count==1 and rag.call_count==2
    assert e.get_experiment(eid)['status']=='completed'
    # A duplicate run cannot re-execute an already completed experiment.
    e.run_experiment(eid);assert rag.call_count==2
    exp2=e.create_experiment(ExperimentCreate(name='Changed',dataset_id='D'))
    monkeypatch.setattr(e,'runtime_manifest',lambda qs,k:{'changed':True})
    e.run_experiment(exp2['experiment_id'])
    assert e.get_experiment(exp2['experiment_id'])['status']=='blocked'
    assert baseline.call_count==1


def test_multi_reviewer_language_aggregation_and_full_export(memory):
    memory.experiments.insert_one({'experiment_id':'E','status':'completed','total_questions':2})
    for q,l,g,hit in [('Q1','en','G1',1),('Q2','hi','G2',0)]:
        memory.experiment_items.insert_one({'experiment_id':'E','question_id':q,'group_id':g,'language':l,'domain':'law','question':'Question', 'rag':answer('rag').model_dump(),'llm_only':answer('llm_only').model_dump(),'retrieval_metrics':{'chunk_hit_at_1':hit,'document_hit_at_3':hit},'answerable_from_kb':True})
        e.save_manual_evaluation('E',q,'rag',ManualEvaluationIn(evaluator='A',correctness=1))
        e.save_manual_evaluation('E',q,'llm_only',ManualEvaluationIn(evaluator='A',correctness=0))
    e.save_manual_evaluation('E','Q1','rag',ManualEvaluationIn(evaluator='B',correctness=0))
    assert memory.evaluations.count_documents({})==5
    result=e.aggregate_experiment('E')
    assert result['by_language']['en']['systems']['rag']['correctness']==.5
    assert result['systems']['rag']['evaluated_count']==2
    assert result['by_language']['hi']['retrieval']['document_hit_at_3']==0
    assert result['paired_comparison']['paired_groups']==2
    assert result['rq3']['miss_at_1']['count']==1
    assert len(e.export_bundle('E')['evaluations'])==5
    assert 'reference_answer' in e.export_csv('E').decode('utf-8-sig')
    with pytest.raises(ValueError):e.save_manual_evaluation('E','MISSING','rag',ManualEvaluationIn(evaluator='A',correctness=1))
    with pytest.raises(ValueError):ManualEvaluationIn(correctness=.5)


def test_rerank_never_mixes_unscored_dense_scores(monkeypatch):
    from app.services import retrieval as r
    emb=Mock();store=Mock();rerank=Mock();rerank.score.return_value=[-4,-2]
    store.search.return_value=[SimpleNamespace(score=.9-i*.01,payload={'chunk_id':str(i),'document_id':'D','text':'Legal text','page_start':1,'page_end':1}) for i in range(4)]
    monkeypatch.setattr(r,'get_embedding_service',lambda:emb);monkeypatch.setattr(r,'get_vector_store',lambda:store);monkeypatch.setattr(r,'get_reranker',lambda:rerank)
    monkeypatch.setattr(r.settings,'rerank_top_n',2)
    result=r.LegalRetriever().retrieve('question',top_k=2)
    assert [x.chunk_id for x in result.results]==['1','0']
    assert len(result.candidates)==4 and result.candidates[2]['rerank_score'] is None

@pytest.mark.parametrize('chunks,expected',[
 (['<thi','nk>private','</thi','nk>Final answer'],'Final answer'),
 (['Okay, let me',' reason','</think>Final'],'Final'),
 (['A normal ','answer'],'A normal answer')])
def test_reasoning_filter(chunks,expected):assert ''.join(visible_tokens(chunks))==expected

def test_unfinished_reasoning_fails():
    with pytest.raises(LLMError):list(visible_tokens(['<think>unfinished']))

def test_ollama_output_limit_fails(monkeypatch):
    from app.services.llm import ollama_client as o
    import httpx
    original=httpx.Client
    monkeypatch.setattr(o.httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(lambda r:httpx.Response(200,text=json.dumps({'message':{'content':'partial'},'done':True,'done_reason':'length'}))),**kw))
    with pytest.raises(LLMError,match='incomplete'):o.OllamaClient().generate('s','q')


def test_constitution_wrapped_heading_footnote_and_schedule():
    text='PART III\n12. Definition.—The State includes authorities.\n13. Laws inconsistent with fundamental\nrights.—All laws shall comply.\n1. Subs. by an Amendment Act.\nSEVENTH SCHEDULE\n45. Land revenue and assessment.'
    chunks=chunk_document('D',[PageText(page_number=1,text=text)],LegalMetadata(document_type='constitution'))
    assert {'12','13'}.issubset({c.article for c in chunks})
    assert not any(c.section in {'1','45'} or c.article in {'1','45'} for c in chunks)
    assert any(c.schedule=='SEVENTH SCHEDULE' for c in chunks)


def test_reindex_bypasses_duplicate_and_preserves_sidecar(tmp_path,monkeypatch):
    from app.services import ingestion as i
    memory=mongomock.MongoClient().db;monkeypatch.setattr(i,'db',memory)
    p=tmp_path/'D.pdf';p.write_bytes(b'fake pdf');p.with_suffix('.metadata.json').write_text('{}')
    memory.documents.insert_one({'document_id':'D','document_hash':i.sha256_file(p),'stored_path':str(p),'ingestion_status':'indexed','metadata':{'title':'Reviewed title'}})
    monkeypatch.setattr(i.settings,'upload_dir',tmp_path)
    monkeypatch.setattr(i,'extract_pdf_pages',lambda p:[]);monkeypatch.setattr(i,'needs_ocr',lambda pages:False)
    index=Mock(return_value={'status':'indexed'});monkeypatch.setattr(i,'_index',index)
    assert i.reindex_document('D')['status']=='indexed'
    assert index.call_count==1
    assert json.loads(p.with_suffix('.metadata.json').read_text())['title']=='Reviewed title'


def test_language_labels_not_just_roman_script():
    assert language_issues('What are the fundamental rights?','hinglish')
    assert language_issues('What are the fundamental rights?','hi')
    assert not language_issues('Bharat ke samvidhan mein kya adhikar hain?','hinglish')
    assert not language_issues('भारत के संविधान में क्या अधिकार हैं?','hi')


def test_calibration_requires_review_and_both_classes():
    with pytest.raises(ValueError):sweep([{'dense_score':.8,'relevant':None}])
    result=sweep([{'dense_score':.8,'relevant':True},{'dense_score':.3,'relevant':False}])
    assert result['selected']['f1']==1
    assert .3<result['recommended_dense_min_score']<=.8


def test_translation_group_gold_mismatch_is_rejected():
    from app.services.datasets import validate_questions
    from app.schemas.dataset import DatasetQuestionIn
    rows=[DatasetQuestionIn(question_id='Q'+l,group_id='G',language=l,domain='law',question='Question text',gold_chunk_ids=[l]) for l in ('en','hi')]
    assert not validate_questions(rows).valid


def test_reviewed_english_translation_and_recheck(tmp_path,monkeypatch):
    from scripts import synthetic_dataset as s
    from test_synthetic_dataset import bundle,draft,verdict
    pack,decisions,sources=bundle()
    pack['config'].update(languages=['en'],model='local',verifier_model='local',timeout=60,tokens=1800,num_ctx=8192,num_gpu=None)
    pack['fingerprint']=s.digest({k:v for k,v in pack.items() if k!='fingerprint'});decisions['fingerprint']=pack['fingerprint']
    run=tmp_path/'base';run.mkdir();s.write_json(run/'candidates.json',pack);s.write_json(run/'decisions.json',decisions)
    hi=draft().model_copy(update={'question':'आवेदन कितने दिनों में दाखिल करना आवश्यक है?','reference_answer':'आवेदन तीस दिनों के भीतर दाखिल करना आवश्यक है।'})
    hinglish=draft().model_copy(update={'question':'Application ko kitne din mein file karna hai?','reference_answer':'Application ko tees din ke andar file karna hai.'})
    llm=Mock();llm.ask.side_effect=[verdict(),hi,verdict(),hinglish,verdict()]
    monkeypatch.setattr(s,'Ollama',lambda *a:llm);monkeypatch.setattr(s,'load_sources',lambda *a:sources)
    args=SimpleNamespace(command='translate',run=run,decisions=run/'decisions.json',output=tmp_path/'translated')
    settings=SimpleNamespace(ollama_base_url='http://localhost:11434')
    assert s.review_transition(args,object(),settings)==0
    out=s.load_json(args.output/'candidates.json');assert len(out['candidates'])==3
    assert len({c['row']['group_id'] for c in out['candidates']})==1
    approvals={'run_id':out['run_id'],'fingerprint':out['fingerprint'],'reviewer':'R','decisions':[{'question_id':c['row']['question_id'],'status':'approve','additional_gold_chunk_ids':[]} for c in out['candidates']]}
    rows,audit=s.export_rows(out,approvals,sources)
    assert len(rows)==3 and all(r['gold_chunk_hashes'] for r in rows)
    approvals['decisions'][0]['edits']={'reference_answer':'changed'}
    with pytest.raises(ValueError,match='recheck'):s.export_rows(out,approvals,sources)
    decisions['decisions'][0]['edits']={'question':'What is the deadline for filing the application?'}
    s.write_json(run/'decisions.json',decisions)
    llm.ask.side_effect=None;llm.ask.return_value=verdict()
    args.command='recheck';args.output=tmp_path/'checked'
    assert s.review_transition(args,object(),settings)==0
    new=s.load_json(args.output/'candidates.json')
    assert new['fingerprint']!=pack['fingerprint'] and new['candidates'][0]['row']['question'].startswith('What is the deadline')
    with pytest.raises(ValueError):s.export_rows(new,decisions,sources)
