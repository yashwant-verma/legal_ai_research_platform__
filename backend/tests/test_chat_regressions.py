"""Regression checks use fake Ollama/retrieval/storage, never local user data."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.chat import router
from app.api.routes.system import config
from app.schemas.retrieval import RetrievalResponse, RetrievedEvidence
from app.services import chat
from app.services.experiments import config_snapshot
from app.services.llm import ollama_client
from app.services.llm.base import LLMError


@pytest.fixture
def storage(monkeypatch):
    runs = Mock()
    monkeypatch.setattr(chat, 'db', SimpleNamespace(research_runs=runs))
    return runs


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def fake_ollama(monkeypatch, handler):
    real_client = httpx.Client
    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(ollama_client.httpx, 'Client', client)
    return ollama_client.OllamaClient()


def test_ollama_stream_payload_and_answer(monkeypatch):
    def handle(request):
        body = json.loads(request.content)
        assert body['stream'] is True
        assert body['think'] is False
        assert body['model'] == ollama_client.settings.ollama_model
        assert body['keep_alive'] == '10m'
        assert body['options']['num_predict'] == 1200
        return httpx.Response(200, text='\n'.join(json.dumps(x) for x in [
            {'message': {'thinking': 'private scratch'}},
            {'message': {'content': 'नमस्ते '}},
            {'message': {'content': 'world'}, 'done': True},
        ]))
    client = fake_ollama(monkeypatch, handle)
    assert client.generate('system', 'query') == 'नमस्ते world'


@pytest.mark.parametrize('response, message', [
    (httpx.Response(404), 'ollama pull'),
    (httpx.Response(400, json={'error': 'bad setting'}), 'bad setting'),
    (httpx.Response(200, text='{"message":{"thinking":"only thinking"},"done":true}\n'), 'no answer text'),
    (httpx.Response(200, text='{"message":{"content":"partial"}}\n'), 'before completing'),
    (httpx.Response(200, text='{"error":"out of memory"}\n'), 'out of memory'),
    (httpx.Response(200, text='not json\n'), 'invalid response'),
])
def test_ollama_failure_is_explicit(monkeypatch, response, message):
    client = fake_ollama(monkeypatch, lambda request: response)
    with pytest.raises(LLMError, match=message):
        client.generate('system', 'query')


@pytest.mark.parametrize('error,code', [(httpx.ReadTimeout('slow'), 504), (httpx.ConnectError('offline'), 503)])
def test_ollama_connection_errors(monkeypatch, error, code):
    def handle(request):
        raise error
    client = fake_ollama(monkeypatch, handle)
    with pytest.raises(LLMError) as exc:
        client.generate('system', 'query')
    assert exc.value.status_code == code


def test_ollama_generation_deadline(monkeypatch):
    client = fake_ollama(monkeypatch, lambda request: httpx.Response(200, text='{"message":{"content":"late"},"done":true}\n'))
    times = iter([0, 301])
    monkeypatch.setattr(ollama_client, 'monotonic', lambda: next(times))
    with pytest.raises(LLMError, match='time limit'):
        client.generate('system', 'query')


def install_llm(monkeypatch):
    llm = Mock()
    llm.stream.side_effect = lambda system, query: iter(['Answer ', 'text [S1].'])
    monkeypatch.setattr(chat, 'get_llm_client', lambda: llm)
    return llm


def test_llm_only_never_retrieves_and_keeps_json_contract(monkeypatch, storage, api):
    llm = install_llm(monkeypatch)
    retrieval = Mock(side_effect=AssertionError('LLM branch invoked retrieval'))
    monkeypatch.setattr(chat, 'get_retriever', retrieval)
    response = api.post('/chat/llm', json={'query': 'test query'})
    assert response.status_code == 200
    answer = response.json()
    assert answer['system_type'] == 'llm_only'
    assert answer['evidence'] == []
    assert answer['answer'] == 'Answer text [S1].'
    retrieval.assert_not_called()
    llm.stream.assert_called_once_with(chat.LLM_ONLY_SYSTEM, 'test query')
    assert storage.insert_one.call_args.args[0]['system_type'] == 'llm_only'


def test_llm_stream_ends_with_logged_answer(monkeypatch, storage, api):
    install_llm(monkeypatch)
    response = api.post('/chat/llm/stream', json={'query': 'test query'})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [e['type'] for e in events] == ['delta', 'delta', 'answer']
    assert events[-1]['answer']['system_type'] == 'llm_only'
    storage.insert_one.assert_called_once()


def test_rag_abstains_without_calling_ollama(monkeypatch, storage, api):
    llm = Mock(side_effect=AssertionError('Abstention invoked Ollama'))
    monkeypatch.setattr(chat, 'get_llm_client', llm)
    retriever = Mock()
    retriever.retrieve.return_value = RetrievalResponse(query='test query', retrieval_status='insufficient_evidence', threshold_used=0.48, candidate_count=0, accepted_count=0, results=[])
    monkeypatch.setattr(chat, 'get_retriever', lambda: retriever)
    response = api.post('/chat/rag/stream', json={'query': 'test query'})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert len(events) == 1
    assert events[0]['answer']['abstained'] is True
    assert events[0]['answer']['answer'] == chat.ABSTENTION_MESSAGE
    llm.assert_not_called()
    storage.insert_one.assert_called_once()


def test_rag_keeps_evidence_prompt_filters_and_top_k(monkeypatch, storage):
    llm = install_llm(monkeypatch)
    evidence = RetrievedEvidence(rank=1, chunk_id='C1', document_id='D1', title='Example', section='2', page_start=3, page_end=3, dense_score=0.9, accepted_as_evidence=True, text='Supplied legal text.')
    retriever = Mock()
    retriever.retrieve.return_value = RetrievalResponse(query='query', retrieval_status='evidence_found', threshold_used=0.48, candidate_count=1, accepted_count=1, results=[evidence])
    monkeypatch.setattr(chat, 'get_retriever', lambda: retriever)
    from app.schemas.retrieval import RetrievalFilters
    filters = RetrievalFilters(official_only=True)
    answer = chat.run_rag('query', filters, top_k=3)
    assert answer.system_type == 'rag'
    assert answer.evidence == [evidence]
    assert not answer.abstained
    retriever.retrieve.assert_called_once_with(query='query', top_k=3, filters=filters)
    system, prompt = llm.stream.call_args.args
    assert system == chat.RAG_SYSTEM
    for text in ['[S1]', 'Page 3', 'Section 2', 'Supplied legal text.']:
        assert text in prompt
    assert storage.insert_one.call_args.args[0]['retrieval']['results'][0]['chunk_id'] == 'C1'


def test_stream_error_is_not_a_success(monkeypatch, storage, api):
    llm = Mock()
    llm.stream.side_effect = LLMError('Ollama timed out', 504)
    monkeypatch.setattr(chat, 'get_llm_client', lambda: llm)
    response = api.post('/chat/llm/stream', json={'query': 'test query'})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events == [{'type': 'error', 'detail': 'Ollama timed out'}]
    storage.insert_one.assert_not_called()


def test_dashboard_and_experiments_use_ollama_model():
    for result in [config(), config_snapshot(5)]:
        assert result['model'] == ollama_client.settings.ollama_model
        assert result['llm_provider'] == 'ollama'
        assert result['ollama_think'] is False


def test_empty_retrieval_does_not_load_reranker(monkeypatch):
    from app.services import retrieval
    embedding, store = Mock(), Mock()
    embedding.embed_query.return_value = [0.1, 0.2]
    store.search.return_value = []
    monkeypatch.setattr(retrieval, 'get_embedding_service', lambda: embedding)
    monkeypatch.setattr(retrieval, 'get_vector_store', lambda: store)
    reranker = Mock(side_effect=AssertionError('No evidence needs no reranker'))
    monkeypatch.setattr(retrieval, 'get_reranker', reranker)
    result = retrieval.LegalRetriever().retrieve('test query')
    assert result.retrieval_status == 'insufficient_evidence'
    reranker.assert_not_called()


def test_real_app_maps_ollama_error_to_json(monkeypatch):
    from app.main import app
    llm = Mock()
    llm.stream.side_effect = LLMError('Ollama is offline', 503)
    monkeypatch.setattr(chat, 'get_llm_client', lambda: llm)
    response = TestClient(app).post('/api/v1/chat/llm', json={'query': 'test query'})
    assert response.status_code == 503
    assert response.json() == {'detail': 'Ollama is offline'}


def test_compare_keeps_both_branches(monkeypatch, storage, api):
    install_llm(monkeypatch)
    retriever = Mock()
    retriever.retrieve.return_value = RetrievalResponse(query='test query', retrieval_status='insufficient_evidence', threshold_used=0.48, candidate_count=0, accepted_count=0, results=[])
    monkeypatch.setattr(chat, 'get_retriever', lambda: retriever)
    response = api.post('/chat/compare', json={'query': 'test query'})
    assert response.status_code == 200
    assert response.json()['llm_only']['system_type'] == 'llm_only'
    assert response.json()['rag']['system_type'] == 'rag'
    assert response.json()['rag']['abstained'] is True
    assert storage.insert_one.call_count == 2


def test_ollama_yields_before_complete_and_closes_on_cancel(monkeypatch):
    class Tokens(httpx.SyncByteStream):
        finished = False
        closed = False
        def __iter__(self):
            yield b'{"message":{"content":"first"}}\n'
            self.finished = True
            yield b'{"message":{"content":"last"},"done":true}\n'
        def close(self):
            self.closed = True
    tokens = Tokens()
    client = fake_ollama(monkeypatch, lambda request: httpx.Response(200, stream=tokens))
    stream = client.stream('system', 'query')
    assert next(stream) == 'first'
    assert not tokens.finished
    stream.close()
    assert tokens.closed
