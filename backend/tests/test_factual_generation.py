import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
from scripts import factual_dataset as g
from test_factual_benchmark import TEXT,rows


def source():
 return {'chunk_id':'C1','document_id':'D1','text':TEXT,'text_sha256':g.digest(TEXT),
         'page_start':1,'page_end':1,'section':'12','document_metadata':{'act_id':'ICA_1872','title':'Indian Contract Act, 1872','version':'Test edition','domain':'contract_law','language':'en','metadata_review_required':False,'status':'in_force'}}

def english():
 return g.Draft(question=rows()[0]['question'],reference_answer='Section 12.',answer_type='provision',provision_type='section',values=['12'],supporting_quote=TEXT)

def args(out,resume=False):
 return SimpleNamespace(domain=['contract_law'],max_source_chars=6500,seed=42,model='fixture-model',output=out,
     resume=resume,groups_per_domain=1,dev_per_domain=0,max_attempts_per_domain=3,retries=1)

def test_generation_resumes_completed_languages_and_exports_drafts(tmp_path,monkeypatch):
 src=source();monkeypatch.setenv('GEMINI_API_KEY','test-key')
 monkeypatch.setattr(g,'prepare_sources',lambda *a:([src],[src]))
 monkeypatch.setattr(g.base,'load_sources',lambda *a:[src])
 calls=[];interrupted=True
 class Fake:
  def __init__(self,*a):self.calls=[]
  def close(self):pass
  def ask(self,model,schema,system,prompt):
   calls.append(schema.__name__)
   if schema==g.Draft:return english()
   if interrupted:raise RuntimeError('Quota unavailable')
   lang='hi' if 'Devanagari' in system else 'hinglish'
   return g.Translation(question=next(q['question'] for q in rows() if q['language']==lang),reference_answer='धारा 12' if lang=='hi' else 'Dhara 12',short_answer_aliases=[])
 monkeypatch.setattr(g,'Gemini',Fake)
 with pytest.raises(RuntimeError,match='Quota'):g.generate(args(tmp_path/'run'),object())
 state=g.load_state(tmp_path/'run/factual_run.json');assert 'en' in state['pending']['C1']['translations']
 interrupted=False;assert g.generate(args(tmp_path/'run',True),object())==0
 assert calls.count('Draft')==1
 candidates=json.loads((tmp_path/'run/candidate_dataset.json').read_text())
 assert len(candidates)==3 and all(q['factual']['review_status']=='draft' for q in candidates)
 assert {q['language'] for q in candidates}=={'en','hi','hinglish'}
 assert len({q['group_id'] for q in candidates})==1
 html=(tmp_path/'run/review.html').read_text();assert '__PACK__' not in html and 'Download reviewed dataset' in html


def test_changed_run_configuration_is_rejected(tmp_path,monkeypatch):
 src=source();monkeypatch.setattr(g,'prepare_sources',lambda *a:([src],[src]))
 out=tmp_path/'run';g.write_json(out/'factual_run.json',{'config':{'wrong':'snapshot'},'revision':1})
 with pytest.raises(ValueError,match='changed'):g.generate(args(out,True),object())


def test_windows_checkpoint_recovers_without_truncating_original(tmp_path,monkeypatch):
 path=tmp_path/'state.json';g.write_json(path,{'revision':1,'value':'old'})
 original=g.os.replace
 def locked(src,dst):
  if Path(dst)==path:raise PermissionError('locked')
  return original(src,dst)
 monkeypatch.setattr(g.os,'replace',locked);monkeypatch.setattr(g.time,'sleep',lambda n:None)
 with pytest.raises(RuntimeError,match='locked'):g.write_json(path,{'revision':2,'value':'new'})
 assert json.loads(path.read_text())['value']=='old'
 assert g.load_state(path)['value']=='new'


def test_gemini_reports_non_json_and_does_not_log_secret(monkeypatch):
 client=g.Gemini('secret-fixture-key');client.client.close()
 client.client=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(400,text='<html>Bad request</html>',headers={'content-type':'text/html'})))
 try:
  with pytest.raises(RuntimeError,match='non-JSON HTTP 400') as error:client.ask('model',g.Draft,'s',{})
  assert 'secret-fixture-key' not in str(error.value)
 finally:client.close()


def test_gemini_structured_request_and_response(monkeypatch):
 def respond(req):
  data=json.loads(req.content);assert data['generationConfig']['responseMimeType']=='application/json'
  assert 'responseJsonSchema' in data['generationConfig']
  return httpx.Response(200,json={'modelVersion':'fixture-v1','candidates':[{'finishReason':'STOP','content':{'parts':[{'text':english().model_dump_json()}]}}]})
 client=g.Gemini('fixture-key');client.client.close();client.client=httpx.Client(transport=httpx.MockTransport(respond))
 try:assert client.ask('model',g.Draft,'s',{}).values==['12']
 finally:client.close()

def test_paper_export_requires_real_completed_run_and_writes_figures(tmp_path):
 from scripts.export_factual_paper import export
 from app.services.factual_experiments import summarize,CONDITIONS
 from app.evaluation.factual import score_answer
 from test_factual_benchmark import label,raw
 fixture=rows()
 for row in fixture:
  row['scores']={c:score_answer(label(),raw()) for c in CONDITIONS};row['timings']={c:1.0 for c in CONDITIONS}
 a=summarize(fixture);a.update(conditions=list(CONDITIONS),valid_for_reporting=True,label_semantics=['source_chunk'],
      by_language={r['language']:summarize([r]) for r in fixture},by_domain={'contract_law':summarize(fixture)},paired_comparisons=[])
 bundle={'experiment':{'evaluation_kind':'factual','status':'completed','aggregate':a,'manifest':{'fixture':True}},'items':fixture}
 export(bundle,tmp_path/'plots')
 assert (tmp_path/'plots/performance.csv').read_text(encoding='utf-8-sig').count('llm_only')==5
 assert (tmp_path/'plots/language_accuracy.png').read_bytes().startswith(b'\x89PNG')
 assert '<svg' in (tmp_path/'plots/accuracy_latency.svg').read_text()
 bundle['experiment']['status']='partial'
 with pytest.raises(ValueError,match='completed'):export(bundle,tmp_path/'bad')
