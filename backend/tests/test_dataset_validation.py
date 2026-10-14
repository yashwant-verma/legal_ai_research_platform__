from app.schemas.dataset import DatasetQuestionIn
from app.services.datasets import validate_questions

def test_missing_gold_warns_and_disables_hit_at_k():
    qs=[DatasetQuestionIn(question_id='Q1',language='en',domain='cyber',question='What legal remedy applies?',reference_answer='x')]
    v=validate_questions(qs)
    assert v.valid
    assert not v.capabilities['retrieval_hit_at_k']
    assert any('Hit@K' in w for w in v.warnings)

def test_multilingual_triplet_is_valid():
    qs=[DatasetQuestionIn(question_id=f'Q1_{lang}',group_id='G1',language=lang,domain='cyber',question='abc',reference_answer='x',gold_document_ids=['D1']) for lang in ('en','hi','hinglish')]
    v=validate_questions(qs)
    assert v.valid
    assert v.capabilities['multilingual_analysis']
