import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from scripts import synthetic_dataset as s
from app.services.datasets import parse_dataset_bytes, rows_to_questions


TEXT = ('Section 5. Every application must include the applicant name and be filed within thirty days. '
        'The office must acknowledge receipt in writing. ')


def source(cid='C_REAL', text=None):
    text = text or TEXT * 2
    return {'chunk_id': cid, 'document_id': 'D_REAL', 'text': text, 'text_sha256': s.digest(text),
            'document_hash': 'HASH', 'chunk_version': 'v1', 'metadata': {'title':'Example procedure', 'status':'in_force'},
            'document_metadata': {}, 'section':'5', 'article':None, 'subsection':None, 'heading':None,
            'paragraph_ids': [], 'page_start': 2, 'page_end': 2}


def draft(question='Within how many days must an application be filed?'):
    return s.Draft(skip=False, skip_reason='', question=question,
                   reference_answer='An application must be filed within thirty days.', difficulty='easy',
                   claims=[s.Claim(claim='The deadline is thirty days.', supporting_quote='be filed within thirty days.')])


def verdict(**kwargs):
    fields={k:True for k in ('answerable','all_claims_supported','answer_complete','language_correct','translation_equivalent')}
    fields.update(kwargs)
    return s.Verdict(**fields, reason='Checked against the supplied passage.')


def bundle():
    sources=[source(),source('C_OVERLAP'),source('C_IRRELEVANT', 'A different passage without relevant evidence.' * 5)]
    c=s.make_candidate(sources[0],draft(),verdict(),'en','local','local',sources)
    pack={'run_id':'RUN1','config':{'document_ids':['D_REAL']},'candidates':[c],
          'sources':{x['chunk_id']:x for x in sources}}
    pack['fingerprint']=s.digest(pack)
    d={'run_id':'RUN1','fingerprint':pack['fingerprint'],'reviewer':'Reviewer',
       'decisions':[{'question_id':c['row']['question_id'],'status':'approve','notes':'Verified',
                     'additional_gold_chunk_ids':['C_OVERLAP']}]}
    return pack,d,sources


def test_source_ids_are_attached_by_code_and_alternatives_are_not_auto_gold():
    pack,_,_=bundle()
    c=pack['candidates'][0]
    assert c['row']['gold_chunk_ids']==['C_REAL']
    assert c['row']['gold_document_ids']==['D_REAL']
    assert c['alternative_chunk_ids']==['C_OVERLAP']
    assert c['same_model_verification'] is True


def test_validated_export_is_compatible_with_current_importer(tmp_path,monkeypatch):
    pack,decisions,sources=bundle()
    run=tmp_path/'run';run.mkdir()
    s.write_json(run/'candidates.json',pack);s.write_json(tmp_path/'decisions.json',decisions)
    monkeypatch.setattr(s,'load_sources',lambda db,ids:sources)
    args=SimpleNamespace(run=run,decisions=tmp_path/'decisions.json',output=tmp_path/'export')
    assert s.export(args,object())==0
    for name in ('approved_dataset.json','approved_dataset.csv'):
        rows=rows_to_questions(parse_dataset_bytes(name,(args.output/name).read_bytes()))
        assert rows[0].gold_chunk_ids==['C_OVERLAP','C_REAL']
        assert rows[0].reference_answer==draft().reference_answer
        assert rows[0].gold_document_ids==['D_REAL']
    assert (args.output/'review_audit.json').exists()


@pytest.mark.parametrize('fault',['quote','thinking','empty'])
def test_bad_drafts_rejected(fault):
    d=draft()
    if fault=='quote':d.claims[0].supporting_quote='An invented deadline of sixty days.'
    elif fault=='thinking':d.reference_answer='Let me think about this. </think> The answer is thirty days.'
    else:d.question=''
    with pytest.raises(ValueError):s.validate_draft(d,source())


@pytest.mark.parametrize('fault',['different_run','missing_reviewer','pending','unknown_id','changed_source','extra_gold','machine_failed','duplicate_decision','tampered_bundle'])
def test_export_refuses_unreviewed_stale_or_invalid_gold(fault):
    pack,d,sources=bundle()
    if fault=='different_run':d['run_id']='OTHER'
    elif fault=='missing_reviewer':d['reviewer']=' '
    elif fault=='pending':d['decisions'][0]['status']='pending'
    elif fault=='unknown_id':d['decisions'][0]['question_id']='FAKE'
    elif fault=='changed_source':sources[0]=source(text=TEXT+'Amended after review.')
    elif fault=='extra_gold':d['decisions'][0]['additional_gold_chunk_ids']=['C_IRRELEVANT']
    elif fault=='machine_failed':
        pack['candidates'][0]['machine_passed']=False
        pack['fingerprint']=s.digest({k:v for k,v in pack.items() if k!='fingerprint'})
        d['fingerprint']=pack['fingerprint']
    elif fault=='duplicate_decision':d['decisions'].append(d['decisions'][0])
    elif fault=='tampered_bundle':pack['candidates'][0]['row']['gold_document_ids']=['FAKE']
    with pytest.raises(ValueError):s.export_rows(pack,d,sources)


def test_translation_preserves_group_and_uses_requested_language():
    a=s.make_candidate(source(),draft(),verdict(),'en','a','a',[source()])
    b=s.make_candidate(source(),draft(),verdict(),'hi','a','a',[source()])
    assert a['row']['group_id']==b['row']['group_id']
    assert a['row']['question_id']!=b['row']['question_id']


def test_selection_filters_toc_and_duplicate_text_without_truncation():
    pool=[source(),source('COPY'),source('TOC','TABLE OF CONTENTS\n'+TEXT*2),source('LONG',TEXT*100)]
    assert [c['chunk_id'] for c in s.selection(pool,42,6500)]==['C_REAL']


def test_ollama_uses_schema_no_thinking_and_preserves_settings(monkeypatch):
    original=httpx.Client
    def handle(request):
        payload=json.loads(request.content)
        assert payload['think'] is False and payload['truncate'] is False
        assert payload['format']==s.Draft.model_json_schema()
        assert payload['options']['num_gpu']==0
        return httpx.Response(200,json={'done':True,'message':{'content':draft().model_dump_json()}})
    monkeypatch.setattr(s.httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(handle),**kwargs))
    llm=s.Ollama('http://localhost:11434',60,1800,8192,0)
    try:assert llm.ask('local',s.Draft,'system','prompt')==draft()
    finally:llm.close()


def test_checkpoint_review_resume_and_actual_read_only_mongo_shape(tmp_path,monkeypatch):
    raw=source()
    db=Mock()
    db.documents.find.return_value=[{'document_id':'D_REAL','document_hash':'HASH','metadata':{}}]
    db.chunks.find.return_value=[raw]
    fake=Mock()
    fake.ask.side_effect=[draft(),verdict()]
    monkeypatch.setattr(s,'Ollama',lambda *args:fake)
    args=SimpleNamespace(output=tmp_path,document_id=['D_REAL'],count=1,languages=['en'],seed=42,
        max_source_chars=6500,max_attempts=3,tokens=1800,num_ctx=8192,num_gpu=None,domain=None,
        timeout=60,model=None,verifier_model=None,resume=False)
    settings=SimpleNamespace(ollama_model='local',ollama_base_url='http://localhost:11434')
    assert s.generate(args,db,settings)==0
    pack=s.load_json(tmp_path/'candidates.json')
    assert pack['candidates'][0]['row']['gold_chunk_ids']==['C_REAL']
    html=(tmp_path/'review.html').read_text()
    assert '__DATA_JSON__' not in html
    assert 'Download decisions.json' in html
    assert all(call[0] in ('documents.find','chunks.find') for call in db.mock_calls)
    args.resume=True
    assert s.generate(args,db,settings)==0
    assert fake.ask.call_count==2
    args.model='changed'
    with pytest.raises(ValueError,match='changed'):s.generate(args,db,settings)


def test_review_source_html_is_escaped(tmp_path):
    src=source(text=TEXT+' </script><script>alert(1)</script>')
    c=s.make_candidate(src,draft(),verdict(),'en','local','local',[src])
    run={'run_id':'RUN','config':{},'attempts':[{'candidates':[c]}]}
    s.review_bundle(tmp_path,run,[src])
    html=(tmp_path/'review.html').read_text()
    assert '</script><script>alert(1)</script>' not in html
    assert '\\u003c/script\\u003e' in html


def test_multilingual_export_requires_complete_approved_group():
    pack,d,sources=bundle()
    pack['config']['languages']=['en','hi','hinglish']
    pack['fingerprint']=s.digest({k:v for k,v in pack.items() if k!='fingerprint'})
    d['fingerprint']=pack['fingerprint']
    with pytest.raises(ValueError,match='Incomplete approved translation group'):
        s.export_rows(pack,d,sources)


def test_runtime_failure_stops_then_resume_retries_the_failed_source(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'load_sources',lambda db,ids:[source()])
    llm=Mock()
    llm.ask.side_effect=RuntimeError('Ollama offline')
    monkeypatch.setattr(s,'Ollama',lambda *args:llm)
    args=SimpleNamespace(output=tmp_path,document_id=[],count=1,languages=['en'],seed=42,
        max_source_chars=6500,max_attempts=3,tokens=1800,num_ctx=8192,num_gpu=None,domain=None,
        timeout=60,model=None,verifier_model=None,resume=False)
    settings=SimpleNamespace(ollama_model='local',ollama_base_url='http://localhost:11434')
    assert s.generate(args,object(),settings)==2
    assert llm.ask.call_count==1
    llm.ask.side_effect=[draft(),verdict()]
    args.resume=True
    assert s.generate(args,object(),settings)==0
    state=s.load_json(tmp_path/'run.json')
    assert len(state['failed_attempts'])==1
    assert len(state['attempts'])==1
    assert len(s.candidates_from(state))==1
