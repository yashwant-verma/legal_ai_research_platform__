import hashlib,json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import mongomock
import pytest
from app.evaluation.factual import public_task,score_answer
from app.schemas.experiment import ExperimentCreate
from app.services.datasets import validate_questions,rows_to_questions
from app.services.factual_preflight import check_sources
from app.services import factual_experiments as f
TEXT='12. Applications must be sent to the Registrar. The Registrar receives applications under this Act.'
def label(**updates):
 d={'answer_type':'provision','act_id':'TEST_ACT','act_name':'Example Act','source_version':'2026 edition','canonical_answer':{'act_id':'TEST_ACT','provision_type':'section','values':['12']},'accepted_aliases':[],'supporting_quotes':[{'chunk_id':'C1','quote':TEXT}],'gold_label_kind':'source_chunk','split':'test','review_status':'verified','reviewer':'Reviewer','reviewed_at':'2026-09-13'}
 d.update(updates);return d

def rows():
 return [{'question_id':'Q_'+l,'group_id':'G1','language':l,'domain':'contract_law','question':q,'reference_answer':'Section 12','gold_document_ids':['D1'],'gold_chunk_ids':['C1'],'gold_chunk_hashes':{'C1':hashlib.sha256(TEXT.encode()).hexdigest()},'answerable_from_kb':True,'factual':label()} for l,q in [('en','Which section of the Example Act identifies the recipient?'),('hi','उदाहरण अधिनियम की कौन सी धारा आवेदन प्राप्तकर्ता बताती है?'),('hinglish','Example Act ki kaun si dhara application lene wale ko batati hai?')]]
def raw(values=None,act='TEST_ACT',kind='section',**kw):
 return json.dumps({'answer':{'act_id':act,'provision_type':kind,'values':values or ['12']},'abstained':False,**kw})
@pytest.mark.parametrize('text,correct,valid',[(raw(),1,True),(raw(['13'],explanation='Correct value is Section 12'),0,True),(raw(act='OTHER_ACT'),0,True),(raw(kind='article'),0,True),(raw(['12','13']),0,True),('Section 12',0,False),('{"answer": null, "abstained": true}',0,True),('{"answer": null, "abstained": false}',0,False),('{"answer":null,"abstained":true,"abstained":false}',0,False),('```json\n'+raw()+'\n```',1,True),(raw(['१२']),1,True)])
def test_strict_answer_scoring(text,correct,valid):
 s=score_answer(label(),text);assert s['correct']==correct and s['valid_output']==valid

def test_set_and_alias_scoring():
 gold=label(answer_type='provision_set',canonical_answer={'act_id':'TEST_ACT','provision_type':'section','values':['12','13']})
 assert score_answer(gold,raw(['13','12']))['correct']==1
 assert score_answer(gold,raw(['12','13','13']))['correct']==0
 gold=label(answer_type='short_text',canonical_answer={'act_id':'TEST_ACT','provision_type':'none','values':['Registrar']},accepted_aliases=['रजिस्ट्रार'])
 assert score_answer(gold,raw(['रजिस्ट्रार'],kind='none'))['correct']==1
 assert score_answer(gold,raw(['Registrar or judge'],kind='none'))['correct']==0

def test_public_prompt_excludes_private_labels():
 p=public_task(rows()[0]);text=json.dumps(p)
 assert 'C1' not in text and 'D1' not in text and 'canonical_answer' not in text and 'supporting_quote' not in text and 'Section 12' not in text
 assert p['act_id']=='TEST_ACT'

def test_group_validation_and_csv_round_trip():
 qs=rows_to_questions(rows());assert validate_questions(qs).valid
 assert not validate_questions(qs[:2]).valid
 changed=deepcopy(rows());changed[1]['factual']['canonical_answer']['values']=['13']
 assert not validate_questions(rows_to_questions(changed)).valid
 csv_row=deepcopy(rows()[0]);csv_row['factual']=json.dumps(csv_row['factual'])
 assert rows_to_questions([csv_row])[0].factual.canonical_answer.values==['12']

@pytest.fixture
def database():
 d=mongomock.MongoClient().db;m={'act_id':'TEST_ACT','title':'Example Act','version':'2026 edition','domain':'contract_law','metadata_review_required':False,'language':'en','status':'in_force'}
 d.documents.insert_one({'document_id':'D1','metadata':m,'ingestion_status':'indexed'});d.chunks.insert_one({'chunk_id':'C1','document_id':'D1','text':TEXT,'metadata':m});return d

def test_preflight_sources_hashes_quotes_and_version(database):
 assert check_sources(rows(),database,False)['valid']
 bad=rows();bad[0]['factual']['supporting_quotes'][0]['quote']='Invented quote which never appeared.'
 assert not check_sources(bad,database,False)['valid']
 database.chunks.update_one({'chunk_id':'C1'},{'$set':{'text':'Changed source'}})
 assert any('source text changed' in e for e in check_sources(rows(),database,False)['errors'])

def test_preflight_vector_mismatch(database,monkeypatch):
 from app.services import vector_store
 store=Mock();store.verify_chunks.return_value=['C1: absent from Qdrant'];monkeypatch.setattr(vector_store,'get_vector_store',lambda:store)
 result=check_sources(rows(),database,True);assert not result['valid'] and store.verify_chunks.call_count==1

def test_reranking_override_does_not_mutate_settings(monkeypatch):
 from app.services import retrieval as r
 store=Mock();store.search.return_value=[SimpleNamespace(score=.9,payload={'chunk_id':'C1','document_id':'D1','text':TEXT,'page_start':1,'page_end':1})]
 monkeypatch.setattr(r,'get_embedding_service',lambda:Mock());monkeypatch.setattr(r,'get_vector_store',lambda:store)
 ranker=Mock();ranker.score.return_value=[1];factory=Mock(return_value=ranker);monkeypatch.setattr(r,'get_reranker',factory)
 enabled=r.settings.enable_reranker;retriever=r.LegalRetriever();retriever.retrieve('question',rerank=False);factory.assert_not_called()
 retriever.retrieve('question',rerank=True);factory.assert_called_once_with(force=True);assert r.settings.enable_reranker==enabled

def test_three_conditions_resume_and_metrics(database,monkeypatch):
 from app.services import experiments as legacy
 monkeypatch.setattr(f,'db',database);monkeypatch.setattr(legacy,'db',database)
 monkeypatch.setattr(f,'get_questions',lambda ds:list(database.dataset_questions.find({'dataset_id':ds},{'_id':0})))
 monkeypatch.setattr(f,'manifest',lambda qs,k,split:{'config':{'label_semantics':['source_chunk']},'hash':f.fingerprint(qs)})
 monkeypatch.setattr(f,'check_sources',lambda *args:{'valid':True,'errors':[]})
 database.dataset_questions.insert_many([dict(q,dataset_id='DS') for q in rows()]);exp=f.create(ExperimentCreate(name='Test',dataset_id='DS',mode='factual_compare'))
 calls=[];fail=True
 def answer(q,c,k):
  calls.append((q['question_id'],c))
  if fail and q['language']=='en' and c=='rag_rerank':raise RuntimeError('GPU failure')
  trace={'candidates':[{'chunk_id':'C1','document_id':'D1','passed_threshold':True}],'pre_context_results':[{'chunk_id':'C1','document_id':'D1'}],'results':[{'chunk_id':'C1','document_id':'D1'}]}
  return {'answer':raw(),'retrieval_trace':trace if c!='llm_only' else None}
 monkeypatch.setattr(f,'run_factual',answer);f.run(exp['experiment_id']);a=f.aggregate(exp['experiment_id'])
 assert len(calls)==9 and a['systems']['rag_rerank']['factual_accuracy']==pytest.approx(2/3)
 assert a['systems']['rag_rerank']['failure_rate']==pytest.approx(1/3)
 assert a['systems']['rag_rerank']['retrieval']['chunk_hit_at_1']==pytest.approx(2/3)
 assert a['systems']['llm_only']['retrieval'] is None and a['paired_comparisons'][0]['paired_groups']==1
 fail=False;f.run(exp['experiment_id']);assert len(calls)==10
 a=f.aggregate(exp['experiment_id']);assert a['systems']['rag_rerank']['factual_accuracy']==1
 assert a['systems']['rag_rerank']['retry_count']==1 and a['valid_for_reporting']
 f.run(exp['experiment_id']);assert len(calls)==10
 assert b'canonical_answer' in f.export_csv(exp['experiment_id'])
 assert b'language,en' in f.export_csv(exp['experiment_id'],True)

def test_trace_stage_diagnosis():
 q=rows()[0];assert f.trace_diagnosis(q,None)=='retrieval_failed_before_trace'
 assert f.trace_diagnosis(q,{'candidates':[]})=='gold_not_in_dense_candidates'
 trace={'candidates':[{'chunk_id':'C1','passed_threshold':False}]};assert f.trace_diagnosis(q,trace)=='gold_rejected_by_threshold'
 trace['candidates'][0]['passed_threshold']=True;assert f.trace_diagnosis(q,trace)=='gold_dropped_by_ranking_or_top_k'
 trace['pre_context_results']=[{'chunk_id':'C1'}];assert f.trace_diagnosis(q,trace)=='gold_excluded_by_context_budget'

def test_factual_prompt_retrieves_query_only(monkeypatch):
 from app.services import factual_chat as c
 from app.services import retrieval as r
 retriever=Mock();retriever.retrieve.return_value=SimpleNamespace(model_dump=lambda **kw:{'results':[],'candidates':[]})
 monkeypatch.setattr(r,'get_retriever',lambda:retriever);got=c.run_factual(rows()[0],'rag_no_rerank',5)
 assert got['generation_skipped'] and retriever.retrieve.call_args.args==(rows()[0]['question'],)
 assert score_answer(label(),got['answer'])['abstained']

def test_invalid_output_is_not_request_failure():
 r=rows()[0];r['scores']={c:score_answer(label(),'not json') for c in f.CONDITIONS};r['timings']={c:1 for c in f.CONDITIONS}
 a=f.summarize([r])['systems']['llm_only'];assert a['factual_accuracy']==0 and a['invalid_output_rate']==1 and a['failure_rate']==0

def test_vector_payload_verification_uses_actual_points():
 from qdrant_client import QdrantClient,models
 from app.services.vector_store import VectorStore
 import uuid
 store=object.__new__(VectorStore);store.client=QdrantClient(':memory:');store.collection='test'
 store.client.create_collection('test',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
 chunk={'chunk_id':'C1','document_id':'D1','text':TEXT,'metadata':{'act_id':'TEST_ACT'}}
 assert 'absent' in store.verify_chunks([chunk])[0]
 store.client.upsert('test',[models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,'C1')),vector=[1.0,0.0],payload=chunk)])
 assert not store.verify_chunks([chunk])
 assert 'mismatch' in store.verify_chunks([{**chunk,'text':'changed'}])[0]
 store.client.close()


def test_frozen_manifest_blocks_modified_labels(database,monkeypatch):
 from app.services import experiments as legacy
 monkeypatch.setattr(f,'db',database);monkeypatch.setattr(legacy,'db',database)
 database.dataset_questions.insert_many([dict(q,dataset_id='DS') for q in rows()])
 monkeypatch.setattr(f,'get_questions',lambda ds:list(database.dataset_questions.find({'dataset_id':ds},{'_id':0})))
 monkeypatch.setattr(f,'manifest',lambda qs,k,split:{'config':{'label_semantics':['source_chunk']},'hash':f.fingerprint(qs)})
 monkeypatch.setattr(f,'check_sources',lambda *a:{'valid':True,'errors':[]})
 exp=f.create(ExperimentCreate(name='Freeze',dataset_id='DS',mode='factual_compare'))
 database.dataset_questions.update_one({'question_id':'Q_en'},{'$set':{'question':'Changed question'}})
 generate=Mock();monkeypatch.setattr(f,'run_factual',generate);f.run(exp['experiment_id'])
 generate.assert_not_called();assert database.experiments.find_one({'experiment_id':exp['experiment_id']})['status']=='blocked'


def test_development_and_legacy_rows_not_selected_as_test(monkeypatch):
 all_rows=rows()+[{'question_id':'legacy','factual':None}]
 all_rows[0]['factual']['split']='development'
 monkeypatch.setattr(f,'get_questions',lambda ds:all_rows)
 assert len(f.selected_questions('DS','test'))==2
