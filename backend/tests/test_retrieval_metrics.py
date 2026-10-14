from app.evaluation.retrieval_metrics import calculate_hit_at_k


def test_document_hit_example():
    result = calculate_hit_at_k(
        retrieved_document_ids=["DOC_422", "DOC_153", "DOC_812", "DOC_091", "DOC_774"],
        retrieved_chunk_ids=["C1", "C2", "C3", "C4", "C5"],
        gold_document_ids=["DOC_153"],
        gold_chunk_ids=["C2"],
    )
    assert result.document_hit_at_1 == 0
    assert result.document_hit_at_3 == 1
    assert result.document_hit_at_5 == 1
    assert result.chunk_hit_at_1 == 0
    assert result.chunk_hit_at_3 == 1
    assert result.chunk_hit_at_5 == 1


def test_multi_source_recall_is_separate_from_binary_hit():
    result = calculate_hit_at_k(
        retrieved_document_ids=["A", "X", "Y", "Z", "Q"],
        retrieved_chunk_ids=[],
        gold_document_ids=["A", "B"],
        gold_chunk_ids=[],
    )
    assert result.document_hit_at_1 == 1
    assert result.gold_document_recall_at_5 == 0.5
