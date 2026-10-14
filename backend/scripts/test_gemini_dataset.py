"""Offline regression tests; run from backend with pytest (no API key/network needed)."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import httpx
import pytest
import gemini_dataset as g

TEXT = ('Section 5. Every application must include the applicant name and be filed within thirty days. '
        'The office must acknowledge receipt in writing. ') * 2


def source():
    return {'chunk_id': 'C_ACTUAL', 'document_id': 'D_ACTUAL', 'text': TEXT,
            'text_sha256': g.base.digest(TEXT), 'metadata': {}, 'document_metadata': {},
            'page_start': 2, 'page_end': 2}


def draft(lang='en'):
    texts = {
        'en': ('Within how many days must the application be filed?', 'It must be filed within thirty days.'),
        'hi': ('आवेदन कितने दिनों के भीतर दाखिल करना आवश्यक है?', 'आवेदन तीस दिनों के भीतर दाखिल करना आवश्यक है।'),
        'hinglish': ('Application kitne din ke andar file karna hai?', 'Application tees din ke andar file karna hai.'),
    }
    q, a = texts[lang]
    return g.base.Draft(skip=False, skip_reason='', question=q, reference_answer=a, difficulty='easy',
        claims=[g.base.Claim(claim='Thirty days.', supporting_quote='be filed within thirty days.')])


def verdict():
    return g.base.Verdict(answerable=True, all_claims_supported=True, answer_complete=True,
                         language_correct=True, translation_equivalent=True, reason='Supported.')


def setup_run(tmp_path, monkeypatch, bad_hindi=False):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-placeholder')
    monkeypatch.setattr(g.base, 'load_sources', lambda db, ids: [source()])
    fake = Mock(calls=[])
    def ask(model, schema, system, prompt):
        if schema == g.base.Verdict:
            return verdict()
        lang = next(k for k, v in g.base.LANGUAGES.items() if v == json.loads(prompt)['language'])
        return draft('en' if bad_hindi and lang == 'hi' else lang)
    fake.ask.side_effect = ask
    monkeypatch.setattr(g, 'Gemini', lambda key: fake)
    args = SimpleNamespace(model='test-model', verifier_model=None, groups=1, document_id=[],
        output=tmp_path/'run', seed=42, max_source_chars=6500, max_attempts=3,
        retries=2, domain='example', resume=False)
    settings = SimpleNamespace(mongodb_db='test', qdrant_collection='test',
        embedding_model='test', qdrant_local_path=str(tmp_path/'qdrant'))
    return args, settings, fake


def test_complete_triplets_import_and_resume_without_extra_requests(tmp_path, monkeypatch):
    args, settings, fake = setup_run(tmp_path, monkeypatch)
    assert g.generate(args, object(), settings) == 0
    data = g.base.load_json(args.output/'auto_dataset.json')
    parsed = g.rows_to_questions(data)
    assert g.validate_questions(parsed).valid
    assert len(data) == 3
    assert {r['language'] for r in data} == {'en', 'hi', 'hinglish'}
    assert len({r['group_id'] for r in data}) == 1
    assert all(r['gold_chunk_ids'] == ['C_ACTUAL'] for r in data)
    assert all(r['gold_chunk_hashes'] == {'C_ACTUAL': g.sha(TEXT)} for r in data)
    assert all('NOT human verified' in r['notes'] for r in data)
    count = fake.ask.call_count
    args.resume = True
    assert g.generate(args, object(), settings) == 0
    assert fake.ask.call_count == count
    args.model = 'changed'
    with pytest.raises(ValueError, match='changed'):
        g.generate(args, object(), settings)


def test_failed_hindi_does_not_skip_hinglish_or_export_incomplete_group(tmp_path, monkeypatch):
    args, settings, fake = setup_run(tmp_path, monkeypatch, bad_hindi=True)
    assert g.generate(args, object(), settings) == 2
    assert g.base.load_json(args.output/'auto_dataset.json') == []
    requests = [json.loads(c.args[3]) for c in fake.ask.call_args_list if c.args[1] == g.base.Draft]
    assert any(r['language'] == g.base.LANGUAGES['hinglish'] for r in requests)
    assert len(g.base.load_json(args.output/'generation_audit.json')['failures']) == 1


def test_invalid_source_quote_rejected():
    d = draft()
    d.claims[0].supporting_quote = 'The deadline is sixty days.'
    with pytest.raises(ValueError, match='exact source'):
        g.base.validate_draft(d, source(), 'en')


def test_rest_payload_key_header_json_and_truncation(monkeypatch):
    finish = ['STOP', 'MAX_TOKENS']
    def handle(req):
        assert req.headers['x-goog-api-key'] == 'secret-placeholder'
        assert 'secret-placeholder' not in str(req.url)
        payload = json.loads(req.content)
        assert payload['generationConfig']['responseMimeType'] == 'application/json'
        return httpx.Response(200, json={'modelVersion': 'actual-version', 'candidates': [
            {'finishReason': finish.pop(0), 'content': {'parts': [
                {'thought': True, 'text': 'not final output'}, {'text': draft().model_dump_json()}]}}]})
    llm = g.Gemini('secret-placeholder')
    llm.client.close()
    llm.client = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        assert llm.ask('test', g.base.Draft, 'system', 'prompt') == draft()
        with pytest.raises(ValueError, match='MAX_TOKENS'):
            llm.ask('test', g.base.Draft, 'system', 'prompt')
        assert llm.calls[0]['model_version'] == 'actual-version'
    finally:
        llm.close()


def test_http_failure_preserves_previous_complete_groups(tmp_path, monkeypatch):
    args, settings, fake = setup_run(tmp_path, monkeypatch)
    fake.ask.side_effect = RuntimeError('Gemini HTTP 429')
    with pytest.raises(RuntimeError, match='429'):
        g.generate(args, object(), settings)
    assert g.base.load_json(args.output/'gemini_run.json')['attempted'] == []
    assert g.base.load_json(args.output/'auto_dataset.json') == []
